import os

import pytest

from helpers import load_pack, machine_has_no_gl


@pytest.fixture(scope="session")
def pack():
    return load_pack()


@pytest.fixture(scope="session")
def gl_available():
    """Skip rendering tests only when this machine cannot create an OpenGL context.

    Any other error (a broken renderer or an incompatible dependency) makes the
    tests fail instead. Set SMPL_AVATAR_REQUIRE_GL=1 to never skip (GPU CI).
    """
    try:
        from smpl_avatar.render import import_pyrender

        pyrender = import_pyrender()
        renderer = pyrender.OffscreenRenderer(8, 8)
        renderer.delete()
    except Exception as exc:  # pragma: no cover - depends on the machine
        if os.environ.get("SMPL_AVATAR_REQUIRE_GL") == "1" or not machine_has_no_gl(exc):
            raise
        pytest.skip(f"offscreen OpenGL rendering unavailable: {exc}")
    return True


