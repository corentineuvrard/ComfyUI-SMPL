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


# Error texts meaning "no usable OpenGL on this machine" (missing EGL/GL/OSMesa
# library, no display, no suitable context), as opposed to a broken renderer.
_NO_GL_MARKERS = (
    "egl", "opengl", "osmesa", "libgl", "glx", "display",
    "failed to initialize pyglet window", "nosuchconfig",
)


def machine_has_no_gl(exc: BaseException) -> bool:
    if isinstance(exc, (AttributeError, NameError, SyntaxError, TypeError)):
        return False  # programming errors: our code or a dependency's API changed
    text = f"{type(exc).__name__}: {exc}".lower()
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
