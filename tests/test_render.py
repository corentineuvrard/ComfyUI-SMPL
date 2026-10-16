"""Rendering tests. They need an OpenGL context; SMPL tests also need the model files."""

import numpy as np
import pytest

from smpl_avatar import render
from smpl_avatar.render import RendererCache, compute_view, render_mesh
from smpl_avatar.segmentation import NUM_PARTS
from helpers import box_mesh, installed_models

MODELS = installed_models()
needs_models = pytest.mark.skipif(not MODELS, reason="no SMPL / SMPL-H model files installed")
each_model = pytest.mark.parametrize("model_type, gender", MODELS, ids=[f"{t}-{g}" for t, g in MODELS])


@pytest.fixture()
def cache():
    cache = RendererCache()
    yield cache
    cache.release()


def test_box_render_is_exact(gl_available, cache):
    # 1 m x 1 m square face towards the camera, 2 m x 2 m view on 200 x 200 px.
    vertices, faces = box_mesh(size=(1.0, 1.0, 0.5))
    view = compute_view(np.array([[-1.0, -1.0, -0.25], [1.0, 1.0, 0.25]]), 200, 200, padding=0.0)
    result = render_mesh(vertices, faces, 200, 200, view, cache=cache)

    assert result.image.shape == (200, 200, 3) and result.image.dtype == np.float32
    assert result.mask.shape == (200, 200) and set(np.unique(result.mask)) <= {0.0, 1.0}
    assert result.mask.sum() == pytest.approx(100 * 100, rel=0.02)
    np.testing.assert_array_equal(result.mask, (np.rint(result.coverage * 255) >= 127).astype(np.float32))
    # Linear orthographic depth: the front face is 0.25 m in front of the origin.
    front = view.camera_z - 0.25
    inner = result.coverage == 1.0
    np.testing.assert_allclose(result.distance[inner], front, atol=2e-3)
    assert result.depth[inner].min() == pytest.approx(1.0)
    assert (result.depth[result.mask == 0] == 0).all()
    assert (result.image[result.mask == 0] == 0).all()  # black background


def test_soft_mask_is_coverage_and_background_colour(gl_available, cache):
    vertices, faces = box_mesh(size=(1.0, 1.0, 1.0))
    vertices = render.rotate_y(vertices, 30.0)  # slanted edges -> partial coverage
    view = compute_view(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), 128, 128, padding=0.0)
    result = render_mesh(vertices, faces, 128, 128, view, background="white", soft_mask=True, cache=cache)
    np.testing.assert_array_equal(result.mask, result.coverage)
    assert ((result.mask > 0) & (result.mask < 1)).any()
    np.testing.assert_allclose(result.image[result.coverage == 0], 1.0)


def test_part_labels_partition_the_silhouette(gl_available, cache):
    vertices, faces = box_mesh(size=(1.0, 1.0, 1.0))
    vertices = render.rotate_y(vertices, 20.0)
    face_labels = np.where(vertices[faces].mean(axis=1)[:, 0] > 0, 16, 3)
    view = compute_view(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), 96, 96, padding=0.0)
    for soft in (False, True):
        result = render_mesh(vertices, faces, 96, 96, view, face_labels=face_labels, soft_mask=soft, cache=cache)
        assert result.part_masks.shape == (NUM_PARTS, 96, 96)
        np.testing.assert_allclose(result.part_masks.sum(axis=0), result.mask, atol=1e-6)
        present = {int(i) for i in np.flatnonzero(result.part_masks.reshape(NUM_PARTS, -1).max(axis=1))}
        assert present == {3, 16}


def test_half_covered_pixels_are_inside_the_hard_mask(gl_available, cache):
    """A 1 m square on a 2 m view at 50 px: its edges run through pixel centres."""
    vertices, faces = box_mesh(size=(1.0, 1.0, 0.5))
    view = compute_view(np.array([[-1.0, -1.0, -0.25], [1.0, 1.0, 0.25]]), 50, 50, padding=0.0)
    result = render_mesh(vertices, faces, 50, 50, view, cache=cache)
    alpha8 = np.rint(result.coverage * 255).astype(int)
    half = (alpha8 > 64) & (alpha8 < 191)
    assert half.sum() == 96  # the 4 x 24 edge-centre pixels
    assert set(np.unique(alpha8[half])) <= {127, 128}
    assert result.mask[half].all()  # half-covered pixels count as body on every driver
    assert result.mask.sum() == 24 * 24 + 96


def test_renderer_is_reused_until_size_changes(gl_available, cache):
    # The cache under test lives in this process, so bypass $SMPL_AVATAR_RENDER_PROCESS=worker.
    vertices, faces = box_mesh()
    view = compute_view(vertices, 32, 32)
    render_mesh(vertices, faces, 32, 32, view, cache=cache, mode="inprocess")
    render_mesh(vertices, faces, 32, 32, view, cache=cache, mode="inprocess")
    assert cache.creations == 1 and cache.size == (32, 32)
    render_mesh(vertices, faces, 48, 32, compute_view(vertices, 48, 32), cache=cache, mode="inprocess")
    assert cache.creations == 2 and cache.size == (48, 32)


@pytest.mark.parametrize("requested, preloaded, expected", [
    (None, False, "inprocess"),
    (None, True, "worker"),
    ("auto", True, "worker"),
    ("inprocess", True, "inprocess"),
    ("worker", False, "worker"),
    (" Worker ", False, "worker"),
])
def test_choose_mode(requested, preloaded, expected):
    assert render.choose_mode(requested, preloaded) == expected


def test_choose_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="SMPL_AVATAR_RENDER_PROCESS"):
        render.choose_mode("gpu", False)


def test_worker_protocol_in_process(gl_available, cache):
    vertices, faces = box_mesh()
    view = compute_view(vertices, 24, 24)
    labels = np.zeros(len(faces), dtype=np.int64)
    reply = render._unpack(render.handle_request(render.encode_request(vertices, faces, 24, 24, view, labels), cache))
    assert reply["color"].shape == (24, 24, 4) and reply["color"].dtype == np.uint8
    assert reply["raw_depth"].shape == (24, 24) and reply["seg"].shape == (24, 24, 3)
    bad = render._unpack(render.handle_request(render.encode_request(vertices, faces, 24, 24, view, labels[:3]), cache))
    assert "face_labels" in str(bad["error"])


@pytest.fixture()
def worker():
    worker = render.RenderWorker()
    yield worker
    worker.close()


def test_worker_process_matches_in_process(gl_available, cache, worker, monkeypatch):
    monkeypatch.setattr(render, "RENDER_WORKER", worker)
    vertices, faces = box_mesh(size=(1.0, 1.0, 1.0))
    vertices = render.rotate_y(vertices, 25.0)
    face_labels = np.where(vertices[faces].mean(axis=1)[:, 0] > 0, 16, 3)
    view = compute_view(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), 80, 64, padding=0.0)
    args = (vertices, faces, 80, 64, view)
    local = render_mesh(*args, face_labels=face_labels, soft_mask=True, cache=cache, mode="inprocess")
    remote = render_mesh(*args, face_labels=face_labels, soft_mask=True, mode="worker")
    assert worker.running and worker.starts == 1
    # The worker may run on another GPU (hybrid laptops): MSAA resolves can
    # differ by one 8-bit step, so compare with that tolerance.
    np.testing.assert_allclose(remote.coverage, local.coverage, atol=1 / 255)
    np.testing.assert_allclose(remote.part_masks, local.part_masks, atol=1 / 255)
    np.testing.assert_allclose(remote.image, local.image, atol=2 / 255)
    np.testing.assert_allclose(remote.depth, local.depth, atol=1 / 255 + 1e-4)

    # Errors raised in the worker are reported and the worker keeps running.
    with pytest.raises(RuntimeError, match="face_labels"):
        render_mesh(*args, face_labels=face_labels[:5], mode="worker")
    assert worker.running

    # A stopped worker is restarted on the next render.
    worker.close()
    assert not worker.running
    render_mesh(*args, mode="worker")
    assert worker.starts == 2


# ------------------------------------------------------------------ SMPL


@needs_models
@each_model
def test_generate_smpl_avatar(gl_available, tmp_path, model_type, gender):
    from smpl_avatar import body_model, pipeline

    avatar = pipeline.generate(
        model_type, gender, 172.0, 68.0, 256, 384, regressor_path=tmp_path / "reg.npz"
    )
    r = avatar.render
    assert r.image.shape == (384, 256, 3)
    assert r.mask.shape == r.depth.shape == (384, 256)
    assert r.part_masks.shape == (NUM_PARTS, 384, 256)
    np.testing.assert_allclose(r.part_masks.sum(axis=0), r.mask, atol=1e-6)
    assert 0.03 < r.mask.mean() < 0.9
    assert avatar.measured_height_cm == pytest.approx(172.0, abs=1.0)
    assert avatar.measured_weight_kg == pytest.approx(68.0, abs=3.0)
    assert (tmp_path / "reg.npz").is_file()
    assert (model_type, gender) in body_model.cached_keys()

    # Every part is visible from the front in the default T-pose.
    visible = r.part_masks.reshape(NUM_PARTS, -1).max(axis=1)
    assert (visible > 0).sum() >= 20


@needs_models
@each_model
def test_arm_angle_lowers_the_wrists(gl_available, model_type, gender):
    from smpl_avatar import body_model

    body = body_model.load(model_type, gender)
    betas = np.zeros(10)
    _, t_joints = body_model.posed_vertices(body, betas, 0.0)
    _, a_joints = body_model.posed_vertices(body, betas, 45.0)
    left_wrist, right_wrist, left_shoulder = 20, 21, 16
    assert abs(t_joints[left_wrist, 1] - t_joints[left_shoulder, 1]) < 0.1
    assert a_joints[left_wrist, 1] < a_joints[left_shoulder, 1] - 0.2
    assert a_joints[right_wrist, 1] < a_joints[left_shoulder, 1] - 0.2
    assert t_joints[left_shoulder, 0] > 0  # SMPL's left is +x


@needs_models
@each_model
def test_generate_node_outputs(pack, gl_available, tmp_path, monkeypatch, model_type, gender):
    torch = pytest.importorskip("torch")
    # The node uses the pack's default regressor path: keep the fit out of the real models/ folder.
    monkeypatch.setattr(pack.smpl_avatar.paths, "REGRESSOR_PATH", tmp_path / "reg.npz")
    pack.smpl_avatar.pipeline._regressors.clear()
    node = pack.NODE_CLASS_MAPPINGS["SMPLAvatarGenerate"]()
    image, mask, depth, parts = node.generate(model_type, gender, 180.0, 80.0, 128, 192, arm_angle=40.0)
    assert image.shape == depth.shape == (1, 192, 128, 3)
    assert mask.shape == (1, 192, 128)
    assert parts.shape == (NUM_PARTS, 192, 128)
    assert all(t.dtype == torch.float32 for t in (image, mask, depth, parts))
    assert float(image.min()) >= 0 and float(image.max()) <= 1
    union = pack.NODE_CLASS_MAPPINGS["SMPLAvatarPartMaskUnion"]()
    (full,) = union.union(parts, preset="full body")
    assert torch.equal(full, mask)


def _wrapped(inner: BaseException) -> BaseException:
    """The exception import_pyrender() raises when importing the renderer fails."""
    try:
        try:
            raise inner
        except BaseException as original:
            raise ImportError(
                "The renderer could not be loaded (...). Install the node pack requirements; "
                "on headless Linux see docs/TECHNICAL.md, 'OpenGL platform and headless Linux'."
            ) from original
    except ImportError as exc:
        return exc


@pytest.mark.parametrize("exc, skip", [
    (ImportError("Unable to load EGL library", "libEGL.so.1: cannot open shared object file"), True),
    (_wrapped(ImportError("('Unable to load OpenGL library', 'libGL.so.1: cannot open')")), True),
    (ValueError('Failed to initialize Pyglet window with an OpenGL >= 3+ context.'), True),
    (RuntimeError('Cannot connect to "None"'), True),
    # Broken installs or renderers must fail, even behind the wrapper message.
    (_wrapped(ModuleNotFoundError("No module named 'trimesh'")), False),
    (_wrapped(ImportError("import of smpl_avatar._vendor.pyrender.scene halted; None in sys.modules")), False),
    (AttributeError("module 'pyglet' has no attribute 'gl'"), False),  # pyglet 3
    (ImportError("cannot import name 'Scene' from 'smpl_avatar._vendor.pyrender'"), False),
    (ValueError("Unsupported PyOpenGL platform: eglx"), False),
    (TypeError("unexpected keyword argument"), False),
])
def test_only_missing_opengl_skips_rendering_tests(exc, skip):
    from helpers import machine_has_no_gl

    assert machine_has_no_gl(exc) is skip


def test_real_import_failure_is_not_mistaken_for_missing_opengl():
    """End to end: a missing dependency behind import_pyrender() must fail, not skip."""
    import subprocess
    import sys
    import textwrap

    from helpers import REPO

    code = textwrap.dedent(f"""
        import sys
        sys.path[:0] = [{str(REPO)!r}, {str(REPO / "tests")!r}]
        sys.modules["trimesh"] = None
        from helpers import machine_has_no_gl
        from smpl_avatar import render
        try:
            render.import_pyrender()
        except ImportError as exc:
            print(machine_has_no_gl(exc))
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_rgb_only_platform_falls_back_to_depth_coverage(monkeypatch):
    """pyrender's OSMesa backend returns RGB without alpha: coverage comes from depth."""
    view = compute_view(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]), 4, 3)
    raw_depth = np.zeros((3, 4), dtype=np.float32)
    raw_depth[1, 1:3] = 0.9
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[1, 1:3] = 200
    monkeypatch.setattr(render, "rasterize", lambda *args, **kwargs: (rgb, raw_depth, None))
    result = render_mesh(*box_mesh(), 4, 3, view, background="white", mode="inprocess")
    np.testing.assert_array_equal(result.mask, (raw_depth > 0).astype(np.float32))
    np.testing.assert_allclose(result.image[0, 0], 1.0)  # background composited
    np.testing.assert_allclose(result.image[1, 1], 200 / 255, atol=1e-6)


def test_worker_crash_reports_its_exit_code(monkeypatch, tmp_path):
    """A worker that dies mid-request is reported with its real exit code."""
    stub = tmp_path / "crashing_worker.py"
    stub.write_text("import sys\nsys.stdin.buffer.read(8)\nsys.exit(3)\n", encoding="utf-8")
    monkeypatch.setattr(render, "WORKER_SCRIPT", stub)
    worker = render.RenderWorker()
    vertices, faces = box_mesh()
    with pytest.raises(RuntimeError, match=r"exit code 3"):
        worker.rasterize(vertices, faces, 8, 8, compute_view(vertices, 8, 8))
    assert not worker.running
    worker.close()
