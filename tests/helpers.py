"""Plain helpers shared by the tests (importable as ``helpers``)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_pack(name: str = "smpl_avatar_pack_under_test"):
    """Import the repository folder the way ComfyUI imports a custom node."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / "__init__.py", submodule_search_locations=[str(REPO)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Root-cause texts meaning "no usable OpenGL on this machine" (a system OpenGL /
# EGL / OSMesa library that cannot be loaded, no display, no suitable context),
# as opposed to a broken install or renderer, which must fail the tests.
_NO_GL_MARKERS = (
    "unable to load opengl library",
    "unable to load egl library",
    "unable to load osmesa library",
    "libgl.so", "libegl.so", "libosmesa",
    "cannot connect to",  # pyglet NoSuchDisplayException
    "nosuchdisplayexception",
    "nosuchconfigexception",
    "failed to initialize pyglet window",
    "eglerror",
)


def root_cause(exc: BaseException) -> BaseException:
    """Innermost exception of a ``raise ... from ...`` / implicit context chain."""
    seen = set()
    while id(exc) not in seen:
        seen.add(id(exc))
        inner = exc.__cause__ or exc.__context__
        if inner is None:
            break
        exc = inner
    return exc


def machine_has_no_gl(exc: BaseException) -> bool:
    root = root_cause(exc)
    if isinstance(root, (AttributeError, NameError, SyntaxError, TypeError, ModuleNotFoundError)):
        return False  # missing package or changed API: a broken install, not a missing GPU
    text = f"{type(root).__name__}: {root}".lower()
    return any(marker in text for marker in _NO_GL_MARKERS)


def box_mesh(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Closed axis-aligned box with outward-facing triangles."""
    sx, sy, sz = (s / 2.0 for s in size)
    corners = np.array(
        [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=np.float64
    ) + np.asarray(center)
    # corner index = 4*ix + 2*iy + iz
    quads = [
        (0, 1, 3, 2),  # -x
        (4, 6, 7, 5),  # +x
        (0, 4, 5, 1),  # -y
        (2, 3, 7, 6),  # +y
        (0, 2, 6, 4),  # -z
        (1, 5, 7, 3),  # +z
    ]
    faces = []
    for a, b, c, d in quads:
        faces += [(a, b, c), (a, c, d)]
    return corners, np.array(faces, dtype=np.int64)


def installed_models():
    from smpl_avatar import paths

    found = []
    for model_type in ("SMPL", "SMPL-H"):
        for gender in ("male", "female"):
            try:
                paths.find_body_model(model_type, gender)
            except FileNotFoundError:
                continue
            found.append((model_type, gender))
    return found
