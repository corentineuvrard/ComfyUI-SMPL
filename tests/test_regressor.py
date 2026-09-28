import json

import numpy as np
import pytest

from smpl_avatar import regressor
from smpl_avatar.regressor import LinearBetaRegressor, NUM_BETAS
from helpers import box_mesh


def synthetic_body(seed=0):
    """A human-sized box whose first two shape directions change height and girth."""
    vertices, faces = box_mesh(size=(0.30, 1.75, 0.14))
    rng = np.random.default_rng(seed)
    shapedirs = rng.normal(scale=0.002, size=(len(vertices), 3, NUM_BETAS))
    shapedirs[:, :, 0] = 0.0
    shapedirs[:, 1, 0] = vertices[:, 1] * 0.06  # beta 0: taller
    shapedirs[:, :, 1] = 0.0
    shapedirs[:, 0, 1] = vertices[:, 0] * 0.12  # beta 1: wider and deeper
    shapedirs[:, 2, 1] = vertices[:, 2] * 0.12
    return vertices, shapedirs, faces


def test_mesh_measurements_on_boxes():
    vertices, faces = box_mesh(size=(2.0, 3.0, 4.0), center=(5.0, -1.0, 2.0))
    assert regressor.mesh_volumes(vertices, faces) == pytest.approx(24.0)
    assert regressor.mesh_heights(vertices) == pytest.approx(3.0)
    # Winding-independent and batched.
    batch = np.stack([vertices, vertices * 2.0])
    np.testing.assert_allclose(regressor.mesh_volumes(batch, faces[:, ::-1]), [24.0, 192.0])
    np.testing.assert_allclose(regressor.mesh_heights(batch), [3.0, 6.0])


def test_measure_betas_uses_density():
    vertices, shapedirs, faces = synthetic_body()
    h, w = regressor.measure_betas(vertices, shapedirs, faces, np.zeros(NUM_BETAS), density=1000.0)
    assert h[0] == pytest.approx(175.0)
    assert w[0] == pytest.approx(0.30 * 1.75 * 0.14 * 1000.0)


def test_predict_shapes():
    reg = LinearBetaRegressor(weights=np.ones((2, NUM_BETAS)), bias=np.arange(NUM_BETAS))
    assert reg.predict(170.0, 60.0).shape == (NUM_BETAS,)
    assert reg.predict([170.0, 180.0], [60.0, 80.0]).shape == (2, NUM_BETAS)
    np.testing.assert_allclose(reg.predict(1.0, 2.0), 3.0 + np.arange(NUM_BETAS))
    assert reg.affine_matrix().shape == (3, NUM_BETAS)


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        LinearBetaRegressor(weights=np.ones((3, NUM_BETAS)), bias=np.zeros(NUM_BETAS))
    with pytest.raises(ValueError):
        LinearBetaRegressor(weights=np.ones((2, NUM_BETAS)), bias=np.zeros(NUM_BETAS + 1))


def test_regressor_key():
    assert regressor.regressor_key("SMPL", "male") == "smpl_male"
    assert regressor.regressor_key("SMPL-H", "female") == "smplh_female"


def test_save_load_roundtrip_without_pickle(tmp_path):
    rng = np.random.default_rng(1)
    regs = {
        "smpl_male": LinearBetaRegressor(rng.normal(size=(2, NUM_BETAS)), rng.normal(size=NUM_BETAS), {"a": 1}),
        "smplh_female": LinearBetaRegressor(rng.normal(size=(2, NUM_BETAS)), rng.normal(size=NUM_BETAS), {"b": [1, 2]}),
    }
    path = tmp_path / "sub" / "reg.npz"
    regressor.save_regressors(path, regs)
    assert not path.with_name("reg.npz.tmp").exists()
    loaded = regressor.load_regressors(path)
    assert set(loaded) == set(regs)
    for key, reg in regs.items():
        np.testing.assert_array_equal(loaded[key].weights, reg.weights)
        np.testing.assert_array_equal(loaded[key].bias, reg.bias)
        assert loaded[key].meta == reg.meta
    with np.load(path, allow_pickle=False) as data:  # plain arrays only
        assert all(data[name].dtype != object for name in data.files)
        assert json.loads(str(data["smpl_male__meta"])) == {"a": 1}


def test_load_rejects_other_format_version(tmp_path):
    path = tmp_path / "reg.npz"
    np.savez(path, format_version=np.array(99), feature_names=np.array(regressor.FEATURE_NAMES), keys=np.array([], dtype=str))
    with pytest.raises(ValueError, match="version"):
        regressor.load_regressors(path)


def test_fit_least_squares_recovers_affine_map():
    rng = np.random.default_rng(2)
    true = LinearBetaRegressor(rng.normal(size=(2, NUM_BETAS)), rng.normal(size=NUM_BETAS))
    h = rng.uniform(150, 200, 500)
    w = rng.uniform(40, 120, 500)
    fitted = regressor.fit_least_squares(h, w, true.predict(h, w))
    np.testing.assert_allclose(fitted.affine_matrix(), true.affine_matrix(), atol=1e-8)


def test_train_on_synthetic_body_hits_targets():
    vertices, shapedirs, faces = synthetic_body()
    grid = regressor.target_grid(np.arange(165.0, 186.0, 5.0), np.arange(20.0, 29.0, 2.0))
    reg = regressor.train(vertices, shapedirs, faces, samples=4000, seed=3, grid=grid)
    assert reg.weights.shape == (2, NUM_BETAS) and reg.bias.shape == (NUM_BETAS,)
    report = reg.meta["report"]
    assert set(report) == {"grid", "grid_bmi_18.5_30", "held_out", "beta_r2_held_out"}
    assert report["grid"]["height_max_cm"] < 1.5
    assert report["grid"]["weight_max_kg"] < 2.5
    assert report["held_out"]["weight_mape_pct"] < 3.0
    assert report["beta_r2_held_out"][0] > 0.9  # height is explained by beta 0
    assert reg.meta["model_fingerprint"] == regressor.model_fingerprint(vertices, shapedirs, faces)
    assert reg.meta["seed"] == 3 and reg.meta["samples"] == 4000

    h, w = regressor.measure_betas(vertices, shapedirs, faces, reg.predict(175.0, 70.0))
    assert h[0] == pytest.approx(175.0, abs=1.5)
    assert w[0] == pytest.approx(70.0, abs=1.5)


def test_training_is_deterministic():
    vertices, shapedirs, faces = synthetic_body()
    a = regressor.train(vertices, shapedirs, faces, samples=1000, seed=5)
    b = regressor.train(vertices, shapedirs, faces, samples=1000, seed=5)
    np.testing.assert_array_equal(a.affine_matrix(), b.affine_matrix())


def test_fingerprint_changes_with_model():
    vertices, shapedirs, faces = synthetic_body()
    other = shapedirs.copy()
    other[0, 0, 0] += 1e-3
    assert regressor.model_fingerprint(vertices, shapedirs, faces) != regressor.model_fingerprint(vertices, other, faces)


def test_volume_polynomial_matches_mesh_volume():
    vertices, shapedirs, faces = synthetic_body(seed=4)
    rng = np.random.default_rng(4)
    shapedirs = shapedirs + rng.normal(scale=0.01, size=shapedirs.shape)  # generic directions
    betas = rng.normal(size=(50, NUM_BETAS)) * 2.0
    template, basis = regressor.shape_basis(vertices, shapedirs)
    meshes = (template + betas @ basis.T).reshape(50, len(vertices), 3)
    direct = regressor.mesh_volumes(meshes, faces)
    np.testing.assert_allclose(regressor.VolumePolynomial(vertices, shapedirs, faces)(betas), direct, rtol=1e-10)
    h, w = regressor.ShapeMeasurer(vertices, shapedirs, faces, density=1000.0)(betas)
    np.testing.assert_allclose(h, regressor.mesh_heights(meshes) * 100.0)
    np.testing.assert_allclose(w, direct * 1000.0, rtol=1e-10)
