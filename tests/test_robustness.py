"""Failure paths a user can hit: corrupt files, hung or chatty workers, bad settings."""

import json
import pickle
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from helpers import REPO, box_mesh
from smpl_avatar import gl_platform, paths, regressor, render, segmentation
from smpl_avatar.render import RenderWorker, compute_view
from synthetic_models import dumps_with_fake_chumpy, mano_dict, model_dict


def load_prepare_models():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_models_rb", REPO / "scripts" / "prepare_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ worker


@pytest.fixture()
def worker(monkeypatch):
    monkeypatch.delenv(render.ENV_RENDER_TIMEOUT, raising=False)
    worker = RenderWorker()
    yield worker
    worker.close()


def _stub_worker(tmp_path, body: str):
    stub = tmp_path / "stub_worker.py"
    stub.write_text("import sys, time\n" + textwrap.dedent(body), encoding="utf-8")
    return stub


def test_worker_timeout_kills_and_reports(monkeypatch, tmp_path, worker):
    """A worker that never answers is killed after the timeout, with a clear message."""
    stub = _stub_worker(tmp_path, """
        sys.stdout.buffer.write(b"SMPLAVW1"); sys.stdout.flush()
        sys.stdin.buffer.read(8)
        time.sleep(60)
    """)
    monkeypatch.setattr(render, "WORKER_SCRIPT", stub)
    monkeypatch.setenv(render.ENV_RENDER_TIMEOUT, "1")
    vertices, faces = box_mesh()
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match=r"did not answer within 1 s"):
        worker.rasterize(vertices, faces, 8, 8, compute_view(vertices, 8, 8))
    assert time.perf_counter() - start < 10
    assert not worker.running


def test_worker_stdout_contamination_is_detected(monkeypatch, tmp_path, worker):
    """Bytes printed before the protocol (a .pth file, sitecustomize) must not hang the render."""
    stub = _stub_worker(tmp_path, """
        print("hello from sitecustomize")
        sys.stdout.flush()
        sys.stdin.buffer.read(8)
        time.sleep(60)
    """)
    monkeypatch.setattr(render, "WORKER_SCRIPT", stub)
    monkeypatch.setenv(render.ENV_RENDER_TIMEOUT, "20")
    vertices, faces = box_mesh()
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match=r"unexpected output on the render worker's stdout"):
        worker.rasterize(vertices, faces, 8, 8, compute_view(vertices, 8, 8))
    assert time.perf_counter() - start < 15
    assert not worker.running


def test_worker_stderr_is_relayed_to_the_log(monkeypatch, tmp_path, worker, caplog):
    stub = _stub_worker(tmp_path, """
        print("[SMPL Avatar] platform chosen by stub", file=sys.stderr)
        print("driver said something", file=sys.stderr)
        sys.stderr.flush()
        sys.stdin.buffer.read(8)
        sys.exit(2)
    """)
    monkeypatch.setattr(render, "WORKER_SCRIPT", stub)
    vertices, faces = box_mesh()
    with caplog.at_level("INFO", logger=render.LOGGER.name):
        with pytest.raises(RuntimeError, match=r"exit code 2"):
            worker.rasterize(vertices, faces, 8, 8, compute_view(vertices, 8, 8))
        deadline = time.time() + 5
        while time.time() < deadline and "driver said something" not in caplog.text:
            time.sleep(0.05)
    assert "render worker: platform chosen by stub" in caplog.text
    assert "render worker: driver said something" in caplog.text


def test_close_does_not_wait_for_a_stuck_render(monkeypatch, tmp_path):
    """close() (the atexit handler) must return even while a render is blocked on a hung worker."""
    import threading

    stub = _stub_worker(tmp_path, """
        sys.stdout.buffer.write(b"SMPLAVW1"); sys.stdout.flush()
        sys.stdin.buffer.read(8)
        time.sleep(120)
    """)
    monkeypatch.setattr(render, "WORKER_SCRIPT", stub)
    monkeypatch.setenv(render.ENV_RENDER_TIMEOUT, "0")  # no timeout: the render hangs for real
    worker = RenderWorker()
    vertices, faces = box_mesh()
    errors = []

    def blocked_render():
        try:
            worker.rasterize(vertices, faces, 8, 8, compute_view(vertices, 8, 8))
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=blocked_render, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline and not worker.running:
        time.sleep(0.05)
    time.sleep(0.5)  # let the render thread block on the reply
    start = time.perf_counter()
    worker.close(timeout=1.0)
    assert time.perf_counter() - start < 5
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors and "render worker stopped" in errors[0]
    assert not worker.running


@pytest.mark.parametrize("value, expected", [("", 300.0), ("45", 45.0), ("0", None), ("-3", None)])
def test_render_timeout_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(render.ENV_RENDER_TIMEOUT, value)
    assert render.render_timeout() == expected


def test_render_timeout_rejects_garbage(monkeypatch):
    monkeypatch.setenv(render.ENV_RENDER_TIMEOUT, "soon")
    with pytest.raises(ValueError, match=render.ENV_RENDER_TIMEOUT):
        render.render_timeout()


def test_inprocess_rendering_from_another_thread_is_refused(gl_available):
    """The OpenGL context is thread-bound: a clear error, not a driver crash."""
    import threading

    cache = render.RendererCache()
    vertices, faces = box_mesh()
    view = compute_view(vertices, 16, 16)
    render.render_mesh(vertices, faces, 16, 16, view, cache=cache, mode="inprocess")
    caught = []

    def other_thread():
        try:
            render.render_mesh(vertices, faces, 16, 16, view, cache=cache, mode="inprocess")
        except RuntimeError as exc:
            caught.append(str(exc))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert caught and "thread that created the OpenGL context" in caught[0]
    cache.release()


# --------------------------------------------------------------- settings


def test_empty_pyopengl_platform_means_unset(monkeypatch):
    monkeypatch.setattr(gl_platform, "_configured", None)
    monkeypatch.setattr(gl_platform.sys, "platform", "win32")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "")
    gl_platform.configure()
    assert "PYOPENGL_PLATFORM" not in gl_platform.os.environ
    env, _ = gl_platform.worker_environment({"PYOPENGL_PLATFORM": "", "DISPLAY": ":0"}, "linux", set())
    assert "PYOPENGL_PLATFORM" not in env


def test_model_dir_entries_tolerate_quotes_and_spaces(monkeypatch, tmp_path):
    folder = tmp_path / "my models é"
    folder.mkdir()
    monkeypatch.setenv(paths.ENV_MODEL_DIR, f'"{folder}"{paths.os.pathsep} ')
    assert paths.env_model_dirs() == [folder]


def test_not_found_message_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_MODEL_DIR, str(tmp_path))
    monkeypatch.setattr(paths, "PACK_MODELS_DIR", tmp_path / "no-pack-models")
    with pytest.raises(paths.BodyModelNotFoundError) as info:
        paths.find_body_model("SMPL-H", "female")
    text = str(info.value)
    assert "mano.is.tue.mpg.de" in text and paths.ENV_MODEL_DIR in text
    assert "prepare_models.py" in text and str(tmp_path) in text


def test_comfyui_extra_model_paths_are_searched(monkeypatch, tmp_path):
    """A body_models entry of extra_model_paths.yaml (registered in folder_paths) is used."""
    import types

    extra = tmp_path / "shared" / "body_models"
    (extra / "smpl").mkdir(parents=True)
    (extra / "smpl" / "SMPL_MALE.pkl").write_bytes(b"x")
    fake = types.ModuleType("folder_paths")
    fake.models_dir = str(tmp_path / "comfy_models")
    fake.folder_names_and_paths = {"body_models": ([str(extra)], set())}
    fake.add_model_folder_path = lambda name, path: fake.folder_names_and_paths.setdefault(name, ([], set()))[0].append(path)
    fake.get_folder_paths = lambda name: list(fake.folder_names_and_paths[name][0])
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    monkeypatch.delenv(paths.ENV_MODEL_DIR, raising=False)
    roots = paths.body_model_roots()
    assert roots[0] == extra and roots[1] == tmp_path / "comfy_models" / "body_models"
    assert paths.find_body_model("SMPL", "male") == extra / "smpl" / "SMPL_MALE.pkl"


# ------------------------------------------------------------ model files


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    from smpl_avatar import body_model

    monkeypatch.setenv(paths.ENV_MODEL_DIR, str(tmp_path))
    body_model.clear_cache()
    yield tmp_path
    body_model.clear_cache()


def test_unconverted_chumpy_file_gives_format_error(model_dir, monkeypatch):
    from smpl_avatar import body_model
    from helpers import block_import

    block_import(monkeypatch, "chumpy")  # same outcome whether or not chumpy is installed
    (model_dir / "smpl").mkdir()
    (model_dir / "smpl" / "SMPL_MALE.pkl").write_bytes(dumps_with_fake_chumpy(model_dict("SMPL")))
    with pytest.raises(body_model.BodyModelFormatError, match="chumpy"):
        body_model.load("SMPL", "male")


def test_smplh_without_mano_gives_format_error(model_dir):
    from smpl_avatar import body_model

    (model_dir / "smplh").mkdir()
    with open(model_dir / "smplh" / "SMPLH_FEMALE.pkl", "wb") as handle:
        pickle.dump(model_dict("SMPL-H"), handle)
    with pytest.raises(body_model.BodyModelFormatError, match="MANO.*prepare_models"):
        body_model.load("SMPL-H", "female")


class _BrokenPackage:
    """Meta path finder for a package that is installed but fails to import
    (chumpy on NumPy >= 1.24: "cannot import name 'bool' from 'numpy'")."""

    def __init__(self, name: str, error: str):
        self.name, self.error = name, error

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util

        if fullname == self.name or fullname.startswith(self.name + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Raised from a frame that belongs to the package, as the real failure is.
        code = compile(f"raise ImportError({self.error!r}, name='numpy')",
                       f"site-packages/{self.name}/__init__.py", "exec")
        exec(code, module.__dict__)


def test_broken_chumpy_install_gives_format_error(model_dir, monkeypatch):
    import sys

    from smpl_avatar import body_model

    for loaded in [m for m in sys.modules if m == "chumpy" or m.startswith("chumpy.")]:
        monkeypatch.delitem(sys.modules, loaded)
    broken = _BrokenPackage("chumpy", "cannot import name 'bool' from 'numpy'")
    monkeypatch.setattr(sys, "meta_path", [broken] + sys.meta_path)
    (model_dir / "smpl").mkdir()
    (model_dir / "smpl" / "SMPL_MALE.pkl").write_bytes(dumps_with_fake_chumpy(model_dict("SMPL")))
    with pytest.raises(body_model.BodyModelFormatError, match="chumpy.*prepare_models") as info:
        body_model.load("SMPL", "male")
    assert "cannot import name 'bool'" in str(info.value)
    assert ("SMPL", "male") not in body_model.cached_keys()


def test_file_from_another_environment_is_not_blamed_on_chumpy(model_dir, monkeypatch):
    """A pickle that references a module missing here (not chumpy / scipy) gets its own advice."""
    import sys
    import types

    from smpl_avatar import body_model

    fake = types.ModuleType("no_such_pkg")
    fake.Thing = type("Thing", (), {"__module__": "no_such_pkg"})
    monkeypatch.setitem(sys.modules, "no_such_pkg", fake)
    data = model_dict("SMPL")
    data["extra"] = fake.Thing()
    payload = pickle.dumps(data, protocol=4)
    monkeypatch.delitem(sys.modules, "no_such_pkg")
    (model_dir / "smpl").mkdir()
    (model_dir / "smpl" / "SMPL_MALE.pkl").write_bytes(payload)
    with pytest.raises(body_model.BodyModelFormatError, match="no_such_pkg.*another Python environment") as info:
        body_model.load("SMPL", "male")
    assert "chumpy / scipy objects" not in str(info.value)


def test_sparse_matrices_without_scipy_give_format_error(model_dir, monkeypatch):
    from smpl_avatar import body_model
    from helpers import block_import

    block_import(monkeypatch, "scipy")
    (model_dir / "smpl").mkdir()
    (model_dir / "smpl" / "SMPL_MALE.pkl").write_bytes(
        dumps_with_fake_chumpy(model_dict("SMPL"), chumpy_keys=(), sparse_keys=("J_regressor",)))
    with pytest.raises(body_model.BodyModelFormatError, match="scipy.*prepare_models"):
        body_model.load("SMPL", "male")


def test_model_of_the_other_type_is_rejected_before_caching(model_dir):
    """An SMPL-H file in the smpl folder loads in smplx and would only fail at render time."""
    from smpl_avatar import body_model

    (model_dir / "smpl").mkdir()
    with open(model_dir / "smpl" / "SMPL_MALE.pkl", "wb") as handle:
        pickle.dump(model_dict("SMPL-H"), handle)
    with pytest.raises(body_model.BodyModelFormatError, match="52 joints.*SMPL-H file"):
        body_model.load("SMPL", "male")
    assert ("SMPL", "male") not in body_model.cached_keys()


@pytest.mark.parametrize("payload", [b"", b"\x00" * 100, b"<html>login</html>", pickle.dumps([1, 2, 3])])
def test_corrupt_model_file_gives_format_error(model_dir, payload):
    from smpl_avatar import body_model

    (model_dir / "smpl").mkdir()
    (model_dir / "smpl" / "SMPL_MALE.pkl").write_bytes(payload)
    with pytest.raises(body_model.BodyModelFormatError, match="could not be read"):
        body_model.load("SMPL", "male")


def test_smplh_npz_never_unpickles(model_dir):
    """An .npz with object arrays must be rejected, never executed."""
    from smpl_avatar import body_model

    marker = model_dir / "pwned.txt"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    (model_dir / "smplh").mkdir()
    np.savez(model_dir / "smplh" / "SMPLH_MALE.npz", v_template=np.zeros((6890, 3)), evil=np.array([Evil()], dtype=object))
    with pytest.raises(body_model.BodyModelFormatError):
        body_model.load("SMPL-H", "male")
    assert not marker.exists()


def test_regressor_file_never_unpickles(tmp_path):
    marker = tmp_path / "pwned.txt"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    path = tmp_path / "reg.npz"
    np.savez(path, format_version=np.array(1), feature_names=np.array(regressor.FEATURE_NAMES),
             keys=np.array(["smpl_male"]), smpl_male__weights=np.zeros((2, 10)), smpl_male__bias=np.zeros(10),
             smpl_male__meta=np.array([Evil()], dtype=object))
    with pytest.raises(ValueError):
        regressor.load_regressors(path)
    assert not marker.exists()


def test_failed_regressor_write_leaves_no_tmp_file(tmp_path, monkeypatch):
    reg = regressor.LinearBetaRegressor(weights=np.zeros((2, 10)), bias=np.zeros(10), meta={})

    def disk_full(handle, **arrays):
        raise OSError("disk full")

    monkeypatch.setattr(regressor.np, "savez", disk_full)
    with pytest.raises(OSError, match="disk full"):
        regressor.save_regressors(tmp_path / "reg.npz", {"smpl_male": reg})
    assert list(tmp_path.iterdir()) == []


def test_save_regressors_is_byte_deterministic(tmp_path):
    rng = np.random.default_rng(0)
    a = regressor.LinearBetaRegressor(rng.normal(size=(2, 10)), rng.normal(size=10), {"k": 1})
    b = regressor.LinearBetaRegressor(rng.normal(size=(2, 10)), rng.normal(size=10), {"k": 2})
    regressor.save_regressors(tmp_path / "x.npz", {"smplh_female": b, "smpl_male": a})
    regressor.save_regressors(tmp_path / "y.npz", {"smpl_male": a, "smplh_female": b})
    assert (tmp_path / "x.npz").read_bytes() == (tmp_path / "y.npz").read_bytes()


# ----------------------------------------------------------- segmentation


@pytest.mark.parametrize("payload", [b"", b"{", b"<html></html>", "{\"hips\": [1]}".encode("utf-16")])
def test_corrupt_segmentation_file_is_reported_with_path(tmp_path, payload):
    path = tmp_path / "seg.json"
    path.write_bytes(payload)
    with pytest.raises(segmentation.SegmentationUnavailableError, match="unreadable") as info:
        segmentation.load_membership(path)
    assert str(path) in str(info.value)


def test_unpinned_segmentation_file_logs_a_warning(tmp_path, caplog):
    from synthetic_models import write_segmentation

    path = tmp_path / "seg.json"
    write_segmentation(path)
    with caplog.at_level("WARNING", logger=segmentation.LOGGER.name):
        membership = segmentation.load_membership(path)
    assert membership.shape == (6890, 24)
    assert "not the pinned SMPL part segmentation" in caplog.text
    assert segmentation.SEGMENTATION_SHA256 in caplog.text


def test_oversized_download_is_rejected(tmp_path, monkeypatch):
    class Huge:
        def read(self, size=-1):
            return b"x" * (size if size and size > 0 else segmentation.MAX_DOWNLOAD_BYTES + 1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(segmentation.urllib.request, "urlopen", lambda url, timeout: Huge())
    with pytest.raises(segmentation.SegmentationUnavailableError, match="larger than"):
        segmentation.download(tmp_path / "seg.json")
    assert not (tmp_path / "seg.json.tmp").exists()


# -------------------------------------------------------------- scripts


def test_prepare_models_reports_corrupt_and_duplicate_files(tmp_path, capsys):
    prepare = load_prepare_models()
    src = tmp_path / "src"
    src.mkdir()
    (src / "basicmodel_m_lbs_10_207_0_v1.1.0.pkl").write_bytes(b"not a pickle at all")
    (src / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl").write_bytes(b"ignored neutral")
    (src / "basicmodel_f_lbs_10_207_0_v1.1.0.pkl").write_bytes(b"corrupt newest")
    (src / "basicmodel_f_lbs_10_207_0_v1.0.0.pkl").write_bytes(dumps_with_fake_chumpy(model_dict("SMPL")))
    assert prepare.main(["--source", str(src), "--output", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "basicmodel_m_lbs_10_207_0_v1.1.0.pkl could not be read as a model file" in out
    assert "no readable SMPL male file" in out
    # The newest version is preferred, and an unreadable one falls back to the next.
    assert "found  SMPL   female" in out and "v1.1.0.pkl" in out.split("also  ")[0]
    assert "basicmodel_f_lbs_10_207_0_v1.1.0.pkl could not be read" in out
    assert "gender-neutral SMPL is not used" in out
    assert (tmp_path / "out" / "smpl" / "SMPL_FEMALE.pkl").is_file()
    assert not (tmp_path / "out" / "smpl" / "SMPL_MALE.pkl").exists()


def test_prepare_models_converts_sparse_matrices_without_scipy(tmp_path):
    """csc (SMPL) and coo (SMPL-H) J_regressor pickles are densified by NumPy alone."""
    prepare = load_prepare_models()
    rng = np.random.default_rng(3)
    dense = np.zeros((24, 6890))
    dense[rng.integers(0, 24, 300), rng.integers(0, 6890, 300)] = rng.uniform(0.1, 1.0, 300)

    def csc_state(m):
        indptr = [0]
        indices, data = [], []
        for col in range(m.shape[1]):
            rows = np.flatnonzero(m[:, col])
            indices.extend(rows)
            data.extend(m[rows, col])
            indptr.append(len(indices))
        return {"format": "csc", "_shape": m.shape, "indptr": np.array(indptr), "indices": np.array(indices),
                "maxprint": 50, "data": np.array(data)}

    rows, cols = np.nonzero(dense)
    states = {
        "scipy.sparse.csc.csc_matrix": csc_state(dense),
        "scipy.sparse._coo.coo_matrix": {"_shape": dense.shape, "data": dense[rows, cols], "coords": (rows, cols),
                                         "has_canonical_format": True, "maxprint": 50},
        "scipy.sparse.coo.coo_matrix": {"_shape": dense.shape, "data": dense[rows, cols], "row": rows, "col": cols,
                                        "maxprint": 50},
    }
    for cls, state in states.items():
        module, name = cls.rsplit(".", 1)
        placeholder = type(name, (prepare.SparsePlaceholder,), {"sparse_class": cls})()
        placeholder.__setstate__(state)
        np.testing.assert_array_equal(placeholder.to_array(), dense)
    with pytest.raises(ValueError, match="unsupported sparse"):
        p = type("dia_matrix", (prepare.SparsePlaceholder,), {"sparse_class": "scipy.sparse.dia_matrix"})()
        p.__setstate__({"_shape": (2, 2), "data": np.zeros(1)})
        p.to_array()


def test_prepare_models_handles_zip_sparse_and_mano_without_scipy(tmp_path):
    """End to end on an official-style zip, in a subprocess where scipy cannot be imported."""
    archive = tmp_path / "SMPL_python_v.1.1.0.zip"
    import zipfile

    smpl = model_dict("SMPL")
    smplh = model_dict("SMPL-H")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
                    dumps_with_fake_chumpy(smpl, sparse_keys=("J_regressor",)))
        zf.writestr("mano/models/SMPLH_female.pkl",
                    dumps_with_fake_chumpy(smplh, sparse_keys=("J_regressor",)))
        zf.writestr("mano/models/MANO_LEFT.pkl", pickle.dumps(mano_dict()))
        zf.writestr("mano/models/MANO_RIGHT.pkl", pickle.dumps(mano_dict()))
    code = textwrap.dedent(f"""
        import sys
        sys.modules["scipy"] = None  # any scipy import now fails
        sys.argv = ["prepare_models.py", "--source", {str(tmp_path)!r}, "--output", {str(tmp_path / "out")!r}]
        sys.path.insert(0, {str(REPO / "scripts")!r})
        import runpy
        runpy.run_path({str(REPO / "scripts" / "prepare_models.py")!r}, run_name="__main__")
    """)
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in (tmp_path / "out").rglob("*.pkl")) == ["SMPLH_FEMALE.pkl", "SMPL_MALE.pkl"]
    with open(tmp_path / "out" / "smplh" / "SMPLH_FEMALE.pkl", "rb") as handle:
        merged = pickle.load(handle)
    assert merged["hands_meanl"].shape == (45,)
    assert type(merged["J_regressor"]) is np.ndarray  # densified without SciPy
    np.testing.assert_array_equal(merged["J_regressor"], smplh["J_regressor"])
    with open(tmp_path / "out" / "smpl" / "SMPL_MALE.pkl", "rb") as handle:
        np.testing.assert_array_equal(pickle.load(handle)["J_regressor"], smpl["J_regressor"])


def test_download_script_fails_cleanly_offline(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("download_segmentation_rb", REPO / "scripts" / "download_segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def offline(url, timeout):
        raise OSError("name resolution failed")

    monkeypatch.setattr(segmentation.urllib.request, "urlopen", offline)
    assert module.main(["--output", str(tmp_path / "seg.json")]) == 1


def test_download_script_checks_an_existing_file(tmp_path, monkeypatch, capsys):
    import importlib.util

    from synthetic_models import write_segmentation

    spec = importlib.util.spec_from_file_location("download_segmentation_rb2", REPO / "scripts" / "download_segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "seg.json"
    write_segmentation(path)  # well formed, but not the pinned file
    assert module.main(["--output", str(path)]) == 1
    out = capsys.readouterr().out
    assert "not the pinned file" in out and "--force" in out

    def offline(url, timeout):
        raise OSError("name resolution failed")

    monkeypatch.setattr(segmentation.urllib.request, "urlopen", offline)
    assert module.main(["--output", str(path), "--force"]) == 1  # --force tries to download
    assert "name resolution failed" in capsys.readouterr().out
    assert path.is_file()  # the old file is untouched by a failed download


def test_train_script_tolerates_unreadable_output(tmp_path, monkeypatch, capsys):
    """An existing --output in another format is replaced with a warning, like the node does."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_regressor_rb", REPO / "scripts" / "train_regressor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bad = tmp_path / "reg.npz"
    np.savez(bad, format_version=np.array(99), feature_names=np.array(regressor.FEATURE_NAMES), keys=np.array([], dtype=str))
    monkeypatch.setenv(paths.ENV_MODEL_DIR, str(tmp_path / "nothing"))
    monkeypatch.setattr(paths, "PACK_MODELS_DIR", tmp_path / "no-pack-models")
    from smpl_avatar import body_model

    body_model.clear_cache()
    # No model installed: the script warns about the file, then reports nothing to do.
    assert module.main(["--output", str(bad), "--model-types", "SMPL", "--genders", "male"]) == 1
    assert "warning: ignoring the existing" in capsys.readouterr().out


# ------------------------------------------------------------- extrapolation


@pytest.mark.parametrize("override", [
    {"height_cm": float("nan")}, {"weight_kg": float("inf")}, {"padding": -0.1}, {"padding": float("nan")},
    {"arm_angle": float("inf")}, {"yaw": float("nan")}, {"view_height_m": float("nan")}, {"view_height_m": 0.0},
])
def test_generate_rejects_bad_inputs_before_loading_a_model(monkeypatch, override):
    from smpl_avatar import pipeline

    monkeypatch.setattr(pipeline.body_model, "load", lambda *a, **k: pytest.fail("a model was loaded"))
    kwargs = dict(model_type="SMPL", gender="male", height_cm=175.0, weight_kg=70.0, width=64, height=96)
    kwargs.update(override)
    with pytest.raises(ValueError, match=next(iter(override))):
        pipeline.generate(**kwargs)


def test_fixed_framing_clipping_is_reported():
    from smpl_avatar import pipeline
    from smpl_avatar.render import compute_view
    from helpers import box_mesh

    vertices, _ = box_mesh(size=(2.0, 1.8, 0.3))  # a 2 m wide, 1.8 m tall body
    assert pipeline.clipping_warning(vertices, compute_view(vertices, 512, 768)) is None  # fit
    wide = compute_view(vertices, 512, 768, framing="fixed", view_height_m=4.0)  # 2.67 m wide
    assert pipeline.clipping_warning(vertices, wide) is None
    narrow = compute_view(vertices, 512, 768, framing="fixed", view_height_m=2.2)  # 1.47 m wide
    message = pipeline.clipping_warning(vertices, narrow)
    assert "does not fit the 1.47 x 2.20 m view" in message
    assert "0.27 m on the left" in message and "0.27 m on the right" in message
    assert "view_height_m" in message and "arm_angle" in message
    short = compute_view(vertices, 512, 768, framing="fixed", view_height_m=1.0)
    assert "at the top" in pipeline.clipping_warning(vertices, short)


def test_extrapolation_warning_logic():
    reg = regressor.LinearBetaRegressor(np.zeros((2, 10)), np.zeros(10), {
        "training_height_range_cm": [150.0, 200.0], "training_weight_range_kg": [40.0, 150.0],
    })
    from smpl_avatar.pipeline import extrapolation_warning

    assert extrapolation_warning(reg, 175, 70, 175.3, 70.5, np.ones(10)) is None
    tall = extrapolation_warning(reg, 220, 165, 221.0, 194.0, np.ones(10) * 5.7)
    assert tall and "beyond the range" in tall and "outside the fitted range" in tall
    assert "194.0 kg" in tall and "|beta| reaches 5.7" in tall
    off = extrapolation_warning(reg, 175, 70, 175.0, 78.0, np.ones(10))
    assert off and "does not match this request exactly" in off and "mesh measures" in off
    # Bounds are rounded outward: a request at the stated bound is inside.
    edge = regressor.LinearBetaRegressor(np.zeros((2, 10)), np.zeros(10), {
        "training_height_range_cm": [135.5, 194.3], "training_weight_range_kg": [24.2, 141.6],
    })
    assert extrapolation_warning(edge, 135, 25, 135.2, 25.5, np.ones(10)) is None


def test_segmentation_json_is_the_pinned_one_or_absent():
    """Guard: the on-disk file used by other tests must be the pinned Meshcapade file."""
    import hashlib

    path = paths.SEGMENTATION_PATH
    if not path.is_file():
        pytest.skip("segmentation not downloaded")
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == segmentation.SEGMENTATION_SHA256
    assert segmentation.parse_segmentation(json.loads(payload)).shape == (6890, 24)


def test_release_from_another_thread_keeps_the_renderer_alive(monkeypatch):
    """pyrender's __del__ calls delete(): dropping the last reference would tear down
    the OpenGL context on the wrong thread (noisy at ComfyUI exit)."""
    import gc
    import threading
    import types

    deleted = []

    class FakeRenderer:
        def __init__(self, width, height):
            pass

        def delete(self):
            deleted.append(threading.get_ident())

        __del__ = delete

    monkeypatch.setattr(render, "import_pyrender", lambda: types.SimpleNamespace(OffscreenRenderer=FakeRenderer))
    monkeypatch.setattr(render, "_log_gl_renderer", lambda: None)
    cache = render.RendererCache()

    def create():
        with cache.renderer(16, 16):
            pass

    thread = threading.Thread(target=create)
    thread.start()
    thread.join()
    cache.release()  # from a thread that did not create the renderer
    gc.collect()
    assert cache.size is None
    assert deleted == []
    render._ORPHANED.clear()


@pytest.mark.parametrize("damage", ["truncated", "garbage"])
def test_damaged_smplh_npz_gives_format_error(model_dir, damage):
    from smpl_avatar import body_model

    (model_dir / "smplh").mkdir()
    target = model_dir / "smplh" / "SMPLH_MALE.npz"
    np.savez(target, v_template=np.zeros((6890, 3)), f=np.zeros((13776, 3), dtype=np.int64))
    payload = target.read_bytes()
    target.write_bytes(payload[: len(payload) // 2] if damage == "truncated" else b"PK\x03\x04garbage")
    with pytest.raises(body_model.BodyModelFormatError, match="could not be read"):
        body_model.load("SMPL-H", "male")


def test_version_key_ignores_folder_and_archive_names():
    prepare = load_prepare_models()
    assert prepare.version_key("downloads/SMPL_python_v.1.1.0.zip!smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl") == (1, 0, 0)
    assert prepare.version_key("C:\\models_v2.0.0\\basicmodel_f_lbs_10_207_0_v1.1.0.pkl") == (1, 1, 0)
    assert prepare.version_key("mano/models/SMPLH_male.pkl") == (0, 0, 0)
