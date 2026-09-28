import pytest

from smpl_avatar import gl_platform


@pytest.mark.parametrize("environ, platform, expected", [
    ({"PYOPENGL_PLATFORM": "osmesa"}, "linux", None),       # user choice is respected
    ({"PYOPENGL_PLATFORM": "egl", "DISPLAY": ":0"}, "linux", None),
    ({}, "linux", "egl"),                                  # headless Linux
    ({"DISPLAY": ":1"}, "linux", None),
    ({"WAYLAND_DISPLAY": "wayland-0"}, "linux", None),
    ({}, "win32", None),
    ({}, "darwin", None),
])
def test_decide(environ, platform, expected):
    value, reason = gl_platform.decide(environ, platform)
    assert value == expected
    assert reason


def test_decide_never_mutates_the_environment():
    environ = {"PYOPENGL_PLATFORM": "osmesa"}
    gl_platform.decide(environ, "linux")
    assert environ == {"PYOPENGL_PLATFORM": "osmesa"}


def headless_linux(monkeypatch):
    """Pretend to run on headless Linux; every change is undone after the test."""
    monkeypatch.setattr(gl_platform, "_configured", None)
    monkeypatch.setattr(gl_platform.sys, "platform", "linux")
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "PYOPENGL_PLATFORM"):
        # setenv first so that monkeypatch records the original state, even when
        # the variable is absent and configure() sets it behind monkeypatch's back.
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_configure_respects_existing_value(monkeypatch):
    headless_linux(monkeypatch)
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    value, _ = gl_platform.configure()
    assert value is None
    assert gl_platform.os.environ["PYOPENGL_PLATFORM"] == "osmesa"


def test_configure_sets_egl_when_headless(monkeypatch):
    headless_linux(monkeypatch)
    monkeypatch.delitem(gl_platform.sys.modules, "OpenGL", raising=False)
    value, _ = gl_platform.configure()
    assert value == "egl"
    assert gl_platform.os.environ["PYOPENGL_PLATFORM"] == "egl"
