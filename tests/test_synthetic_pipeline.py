"""End-to-end tests on synthetic model files (no licensed SMPL data needed)."""

import importlib.util
import pickle
import sys
import zipfile

import numpy as np
import pytest

from helpers import REPO
from synthetic_models import (
    dumps_with_fake_chumpy, ellipsoid, mano_dict, model_dict, write_clean_models, write_segmentation,
)

smplx = pytest.importorskip("smplx")


def load_prepare_models():
    spec = importlib.util.spec_from_file_location("prepare_models", REPO / "scripts" / "prepare_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_mesh_has_smpl_sizes():
    vertices, faces = ellipsoid()
    assert vertices.shape == (6890, 3) and faces.shape == (13776, 3)


def test_prepare_models_from_official_style_zip(tmp_path):
    prepare = load_prepare_models()
    archive = tmp_path / "downloads" / "official.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl", dumps_with_fake_chumpy(model_dict("SMPL")))
        zf.writestr("smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl", dumps_with_fake_chumpy(model_dict("SMPL")))
        zf.writestr("mano/models/SMPLH_male.pkl", dumps_with_fake_chumpy(model_dict("SMPL-H")))
        zf.writestr("mano/models/MANO_LEFT.pkl", dumps_with_fake_chumpy(mano_dict(), chumpy_keys=()))
        zf.writestr("mano/models/MANO_RIGHT.pkl", dumps_with_fake_chumpy(mano_dict(), chumpy_keys=()))
        zf.writestr("__MACOSX/._SMPLH_female.pkl", b"junk")
    assert "chumpy" not in sys.modules

    out = tmp_path / "models"
    assert prepare.main(["--source", str(tmp_path / "downloads"), "--output", str(out)]) == 0
    assert "chumpy" not in sys.modules
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.pkl")) == [
        "smpl/SMPL_FEMALE.pkl", "smpl/SMPL_MALE.pkl", "smplh/SMPLH_MALE.pkl",
    ]

    with open(out / "smpl" / "SMPL_MALE.pkl", "rb") as handle:
        converted = pickle.load(handle)
    reference = model_dict("SMPL")
    for key in ("shapedirs", "v_template", "posedirs", "J_regressor"):
        assert type(converted[key]) is np.ndarray
        np.testing.assert_array_equal(converted[key], reference[key])
    with open(out / "smplh" / "SMPLH_MALE.pkl", "rb") as handle:
        merged = pickle.load(handle)
    np.testing.assert_array_equal(merged["hands_componentsl"], np.eye(45))
    assert merged["hands_meanr"].shape == (45,)

    # smplx loads the converted files without chumpy.
    body = smplx.SMPL(model_path=str(out / "smpl" / "SMPL_MALE.pkl"), gender="male")
    assert body.v_template.shape == (6890, 3)
    smplx.SMPLH(model_path=str(out / "smplh" / "SMPLH_MALE.pkl"), gender="male", use_pca=False)

    # Existing files are kept unless --force is given.
    assert prepare.main(["--source", str(tmp_path / "downloads"), "--output", str(out)]) == 1


def test_prepare_models_rejects_unsupported_chumpy_objects():
    prepare = load_prepare_models()
    placeholder = type("Select", (prepare.ChumpyPlaceholder,), {"chumpy_class": "chumpy.reordering.Select"})()
    placeholder.__setstate__({"a": 1})
    with pytest.raises(ValueError, match="unsupported chumpy object"):
        prepare.to_plain({"x": placeholder})


@pytest.fixture()
def synthetic_install(tmp_path, monkeypatch, pack):
    """Synthetic models + segmentation, wired into both copies of the core package."""
    import smpl_avatar.pipeline  # noqa: F401  (plain copy used by the library tests)

    models = tmp_path / "body_models"
    write_clean_models(models)
    seg = write_segmentation(tmp_path / "seg.json")
    monkeypatch.setenv("SMPL_AVATAR_MODEL_DIR", str(models))
    cores = [sys.modules["smpl_avatar"], pack.smpl_avatar]
    for core in cores:
        monkeypatch.setattr(core.paths, "SEGMENTATION_PATH", seg)
        monkeypatch.setattr(core.paths, "REGRESSOR_PATH", tmp_path / "regressor.npz")
        core.body_model.clear_cache()
        core.pipeline._regressors.clear()
    yield tmp_path
    for core in cores:
        core.body_model.clear_cache()
        core.pipeline._regressors.clear()


def test_body_models_load_lazily_and_are_cached(synthetic_install):
    from smpl_avatar import body_model

    assert body_model.cached_keys() == []
    male = body_model.load("SMPL", "male")
    assert body_model.load("SMPL", "male") is male
    body_model.load("SMPL-H", "female")
    assert body_model.cached_keys() == [("SMPL", "male"), ("SMPL-H", "female")]
    with pytest.raises(ValueError):
        body_model.load("SMPL-X", "male")


def test_pipeline_end_to_end(synthetic_install, gl_available):
    from smpl_avatar import pipeline, regressor

    avatar = pipeline.generate("SMPL-H", "female", 172.0, 70.0, 160, 240)
    r = avatar.render
    assert r.image.shape == (240, 160, 3) and r.part_masks.shape == (24, 240, 160)
    np.testing.assert_allclose(r.part_masks.sum(axis=0), r.mask, atol=1e-6)
    assert avatar.measured_height_cm == pytest.approx(172.0, abs=1.0)
    assert avatar.measured_weight_kg == pytest.approx(70.0, abs=2.0)

    # The regressor was fitted once, saved, and is reused from disk after a restart.
    stored = regressor.load_regressors(synthetic_install / "regressor.npz")
    assert list(stored) == ["smplh_female"]
    assert stored["smplh_female"].meta["model_type"] == "SMPL-H"
    pipeline._regressors.clear()
    again = pipeline.generate("SMPL-H", "female", 172.0, 70.0, 160, 240, part_masks=False)
    np.testing.assert_array_equal(again.betas, avatar.betas)
    assert again.render.part_masks is None


def test_nodes_end_to_end(synthetic_install, gl_available, pack):
    torch = pytest.importorskip("torch")
    generate = pack.NODE_CLASS_MAPPINGS["SMPLAvatarGenerate"]()
    union = pack.NODE_CLASS_MAPPINGS["SMPLAvatarPartMaskUnion"]()
    image, mask, depth, parts = generate.generate("SMPL", "male", 180.0, 75.0, 96, 128, soft_mask=True)
    assert image.shape == depth.shape == (1, 128, 96, 3)
    assert mask.shape == (1, 128, 96) and parts.shape == (24, 128, 96)
    (full,) = union.union(parts, preset="full body")
    torch.testing.assert_close(full, mask)
    (head,) = union.union(parts, preset="head")
    (lower,) = union.union(parts, preset="lower body")
    assert 0 < float(head.sum()) < float(full.sum())
    assert float((head * lower).sum()) == 0.0

    _, _, _, empty = generate.generate("SMPL", "male", 180.0, 75.0, 96, 128, compute_part_masks=False)
    assert empty.shape == (24, 128, 96) and float(empty.abs().sum()) == 0.0
