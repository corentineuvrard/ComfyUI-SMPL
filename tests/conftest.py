import pytest

from helpers import load_pack


@pytest.fixture(scope="session")
def pack():
    return load_pack()


@pytest.fixture(scope="session")
def gl_available():
    """Skip rendering tests when no OpenGL context can be created (e.g. bare CI)."""
    try:
        from smpl_avatar.render import import_pyrender

        pyrender = import_pyrender()
        renderer = pyrender.OffscreenRenderer(8, 8)
        renderer.delete()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"offscreen OpenGL rendering unavailable: {exc}")
    return True


