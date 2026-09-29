# Vendored third-party code

## pyrender

`pyrender/` is a copy of [pyrender](https://github.com/mmatl/pyrender) by
Matthew Matl, MIT license (see `pyrender/LICENSE`), taken from upstream commit
`a59963ef890891656fd17c90e12d663233dcaa99` (2022-04-30, the latest commit;
the latest PyPI release is 0.1.45 from 2021).

Why it is vendored instead of installed from PyPI: pyrender 0.1.45 pins
`PyOpenGL==3.1.0`. Installing it into a current ComfyUI environment downgrades
the `PyOpenGL>=3.1.8` that ComfyUI requires, which breaks ComfyUI's own GLSL
node; PyOpenGL 3.1.0 also has known problems with Python 3.12+ and NumPy 2.
This copy runs with current PyOpenGL releases.

Local changes (the exact diff is `pyrender.patch`; `scripts/vendor_pyrender.py`
re-creates the copy from an upstream checkout):

1. Removed the interactive viewer (`viewer.py`, `trackball.py`) and text
   rendering (`font.py` and the bundled fonts); `Renderer.render_text` raises
   `NotImplementedError`. This drops the pyglet-window, tkinter, imageio and
   freetype-py requirements and the Apache-2.0 font files.
2. Relative imports inside the package, so an installed `pyrender` is never
   mixed in.
3. `np.infty` -> `np.inf` (removed in NumPy 2).
4. Removed the `six` dependency (`abc.ABC` base classes instead of
   `six.add_metaclass`).
5. Fixed the pyglet platform's error path, which referenced the exception
   variable after its `except` block.
