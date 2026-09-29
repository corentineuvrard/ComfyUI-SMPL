"""Guards against dependency conflicts with ComfyUI and with an installed pyrender."""

import re
import subprocess
import sys
import textwrap

import pytest

from helpers import REPO

VENDOR = REPO / "smpl_avatar" / "_vendor" / "pyrender"


def requirement_lines():
    lines = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [line.split("#")[0].strip() for line in lines if line.split("#")[0].strip()]


def pyproject_dependencies():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M).group(1)
    return re.findall(r'"([^"]+)"', block)


def test_requirements_do_not_pull_pypi_pyrender_or_old_pyopengl():
    names = [re.split(r"[<>=!~ ;\[]", line)[0].lower() for line in requirement_lines()]
    assert "pyrender" not in names  # PyPI 0.1.45 pins PyOpenGL==3.1.0
    pyopengl = [line for line in requirement_lines() if line.lower().startswith("pyopengl")]
    assert pyopengl == ["PyOpenGL>=3.1.9"]  # compatible with ComfyUI's PyOpenGL>=3.1.8
    pyglet = [line for line in requirement_lines() if line.lower().startswith("pyglet")]
    assert pyglet == ["pyglet>=1.4.10,<3"]  # pyglet 3 drops pyglet.gl
    for heavy in ("torch", "numpy", "opencv-python", "scikit-learn"):
        assert heavy not in names


def test_pyproject_matches_requirements():
    assert pyproject_dependencies() == requirement_lines()


def test_vendored_pyrender_is_self_contained():
    assert (VENDOR / "LICENSE").is_file()
    assert (VENDOR.parent / "pyrender.patch").is_file()
    for path in VENDOR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import) pyrender\b", source, re.M), path
        assert not re.search(r"^\s*import (six|freetype|imageio)\b", source, re.M), path
        assert "np.infty" not in source, path


def test_renderer_uses_the_vendored_copy():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        from smpl_avatar import render
        try:
            module = render.import_pyrender()
        except ImportError as exc:  # no OpenGL libraries on this machine
            print("SKIP", exc)
            raise SystemExit(0)
        print(module.__name__)
        print("pyrender" in sys.modules)
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    if result.stdout.startswith("SKIP"):
        pytest.skip(result.stdout.strip())
    name, top_level = result.stdout.split()
    assert name == "smpl_avatar._vendor.pyrender"
    assert top_level == "False"
