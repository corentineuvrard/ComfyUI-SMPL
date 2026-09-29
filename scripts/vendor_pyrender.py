"""Re-create smpl_avatar/_vendor/pyrender from an upstream pyrender checkout.

The copy is taken from https://github.com/mmatl/pyrender at commit
a59963ef890891656fd17c90e12d663233dcaa99 (2022-04-30, MIT license) and the
patches below are applied; smpl_avatar/_vendor/pyrender.patch is the
resulting diff. Usage::

    git clone https://github.com/mmatl/pyrender pyrender-upstream
    git -C pyrender-upstream checkout a59963ef890891656fd17c90e12d663233dcaa99
    python scripts/vendor_pyrender.py pyrender-upstream smpl_avatar/_vendor/pyrender
"""

import shutil
import sys
from pathlib import Path

UPSTREAM = Path(sys.argv[1])  # clone of https://github.com/mmatl/pyrender
DEST = Path(sys.argv[2])  # <repo>/smpl_avatar/_vendor/pyrender

SKIP = {"viewer.py", "trackball.py", "font.py", "fonts", "__pycache__"}
TEXT_SUFFIXES = (".py", ".vert", ".frag", ".geom")
CRLF = bytes([13, 10])
LF = bytes([10])

if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)
src = UPSTREAM / "pyrender"
for item in src.iterdir():
    if item.name in SKIP:
        continue
    if item.is_dir():
        shutil.copytree(item, DEST / item.name, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        shutil.copy2(item, DEST / item.name)
shutil.copy2(UPSTREAM / "LICENSE", DEST / "LICENSE")

# Store every text file with LF line endings (the upstream checkout may use CRLF).
for path in DEST.rglob("*"):
    if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == "LICENSE"):
        path.write_bytes(path.read_bytes().replace(CRLF, LF))


def patch(rel, old, new, count=1):
    path = DEST / rel
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{rel}: expected {count} occurrence(s) of {old!r}, found {found}")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


# 1. Package entry: no interactive viewer (pyglet windows, tkinter, imageio).
patch("__init__.py", "from .viewer import Viewer\n", "")
patch("__init__.py", "'Primitive', 'Mesh', 'Node', 'Scene', 'Renderer', 'Viewer',",
      "'Primitive', 'Mesh', 'Node', 'Scene', 'Renderer',")

# 2. Relative imports, so the vendored copy never loads an installed "pyrender".
patch("offscreen.py", "from pyrender.platforms.pyglet_platform import PygletPlatform",
      "from .platforms.pyglet_platform import PygletPlatform")
patch("offscreen.py", "from pyrender.platforms import egl", "from .platforms import egl")
patch("offscreen.py", "from pyrender.platforms.osmesa import OSMesaPlatform",
      "from .platforms.osmesa import OSMesaPlatform")
patch("platforms/pyglet_platform.py", "from pyrender.constants import", "from ..constants import")

# 3. NumPy 2 removed np.infty.
patch("mesh.py", "np.infty", "np.inf", count=6)

# 4. No text rendering (font.py needs freetype-py and bundled Apache-2.0 fonts).
patch("renderer.py", "from .font import FontCache\n", "")
patch("renderer.py", "        self._font_cache = FontCache()\n", "")
patch("renderer.py", """        x *= self.dpscale
        y *= self.dpscale
        font_pt *= self.dpscale

        if color is None:
            color = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            color = format_color_vector(color, 4)

        # Set up viewport for render
        self._configure_forward_pass_viewport(0)

        # Load font
        font = self._font_cache.get_font(font_name, font_pt)
        if not font._in_context():
            font._add_to_context()

        # Load program
        program = self._get_text_program()
        program._bind()

        # Set uniforms
        p = np.eye(4)
        p[0,0] = 2.0 / self.viewport_width
        p[0,3] = -1.0
        p[1,1] = 2.0 / self.viewport_height
        p[1,3] = -1.0
        program.set_uniform('projection', p)
        program.set_uniform('text_color', color)

        # Draw text
        font.render_string(text, x, y, scale, align)
""", """        raise NotImplementedError(
            'Text rendering is not included in this vendored copy of pyrender.'
        )
""")
patch("renderer.py", """        # Free fonts
        self._font_cache.clear()

""", "")

# 5. Drop the "six" dependency (Python 2 compatibility only).
for rel, cls in (("camera.py", "Camera"), ("light.py", "Light"),
                 ("material.py", "Material"), ("platforms/base.py", "Platform")):
    patch(rel, "import six\n", "")
    patch(rel, f"@six.add_metaclass(abc.ABCMeta)\nclass {cls}(object):", f"class {cls}(abc.ABC):")

# 6. Upstream bug: the exception variable is used after its except block
#    (NameError instead of the intended error message when no context works).
patch("platforms/pyglet_platform.py", """        for conf in confs:
            try:""", """        error = None
        for conf in confs:
            try:""")
patch("platforms/pyglet_platform.py", """            except pyglet.window.NoSuchConfigException as e:
                pass""", """            except pyglet.window.NoSuchConfigException as exc:
                error = exc""")
patch("platforms/pyglet_platform.py", """'internal error message was "{}"'.format(e)""",
      """'internal error message was "{}"'.format(error)""")

# 7. Tell this copy apart from the PyPI release in logs and bug reports.
patch("version.py", "__version__ = '0.1.45'", "__version__ = '0.1.45+a59963e.smpl_avatar'")


def write_review_patch():
    """Write ../pyrender.patch: unified diff of upstream -> vendored (for review).

    Modified text files get a unified diff; deleted files are listed with a
    header only. LICENSE is copied verbatim from the upstream root.
    """
    import difflib

    lines = [
        "# Review diff: upstream pyrender a59963ef890891656fd17c90e12d663233dcaa99 -> this copy.\n",
        "# Generated by scripts/vendor_pyrender.py. Deleted files are listed without their\n",
        "# content (not meant for `git apply`). LICENSE is copied verbatim from upstream.\n",
    ]
    for up in sorted(p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = up.relative_to(src).as_posix()
        mine = DEST / rel
        if not mine.exists():
            lines.append(f"deleted: pyrender/{rel}\n")
            continue
        if up.suffix not in TEXT_SUFFIXES:
            continue
        old = up.read_bytes().replace(CRLF, LF).decode("utf-8").splitlines(keepends=True)
        new = mine.read_text(encoding="utf-8").splitlines(keepends=True)
        lines.extend(difflib.unified_diff(old, new, f"a/pyrender/{rel}", f"b/pyrender/{rel}"))
    (DEST.parent / "pyrender.patch").write_text("".join(lines), encoding="utf-8", newline="\n")


write_review_patch()
print("vendored into", DEST)
