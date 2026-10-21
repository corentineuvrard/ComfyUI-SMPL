# Technical notes

Details behind the [README](../README.md): the body-part list, where model
files are searched, how the height/weight regressor is fitted, what the
renderer does, how to run on headless Linux, what was tested, and how to
develop the pack.

## Body parts

`part_masks` holds one mask per SMPL body part, in this order:

| # | Part | # | Part | # | Part |
|---|---|---|---|---|---|
| 0 | `hips` | 8 | `rightFoot` | 16 | `leftArm` (upper arm) |
| 1 | `leftUpLeg` (thigh) | 9 | `spine2` (chest) | 17 | `rightArm` |
| 2 | `rightUpLeg` | 10 | `leftToeBase` | 18 | `leftForeArm` |
| 3 | `spine` (lower torso) | 11 | `rightToeBase` | 19 | `rightForeArm` |
| 4 | `leftLeg` (shin) | 12 | `neck` | 20 | `leftHand` |
| 5 | `rightLeg` | 13 | `leftShoulder` (collar) | 21 | `rightHand` |
| 6 | `spine1` (mid torso) | 14 | `rightShoulder` | 22 | `leftHandIndex1` (fingers) |
| 7 | `leftFoot` | 15 | `head` | 23 | `rightHandIndex1` |

The order is the SMPL joint order: part *i* is the region driven by joint *i*.
"Left" is the body's left.

## Where the model files are searched

1. every folder listed in the environment variable `SMPL_AVATAR_MODEL_DIR`
   (separated by `;` on Windows, `:` elsewhere),
2. every `body_models:` entry of ComfyUI's `extra_model_paths.yaml`, then
   `ComfyUI/models/body_models/`,
3. this pack's `models/` folder.

When ComfyUI loads the pack, it creates `ComfyUI/models/body_models/smpl/`
and `smplh/` if they do not exist yet, each with an empty placeholder file
(`put_SMPL_MALE_and_SMPL_FEMALE_here`, `put_SMPLH_MALE_and_SMPLH_FEMALE_here`),
like ComfyUI's own model folders; existing folders are never touched.

Each location uses the `smpl/` + `smplh/` layout shown in the README (folder and file names
are matched case-insensitively; `smpl-h/` and `smpl_h/` are accepted too;
SMPL-H files may also be `.npz` archives of plain arrays, loaded without
pickle). Models are loaded lazily on first use and cached per (model type,
gender). The height/weight regressor and the part segmentation always live
in this pack's own `models/regressor/` and `models/segmentation/`, next to
the code, whatever folder the body models come from (each installed copy of
the pack fits and downloads them once).

## Preparing the model files

The official downloads cannot be loaded by `smplx` on a current Python /
NumPy stack: they contain `chumpy` objects (chumpy's PyPI release, 0.70,
needs NumPy < 1.24), and the SMPL-H files lack the MANO hand parameters
`smplx` reads from the model file (`hands_componentsl/r`, `hands_meanl/r`).
The `smplx` tools for both steps (`tools/clean_ch.py`, a Python 2 script,
and `tools/merge_smplh_mano.py`) need chumpy themselves.

`scripts/prepare_models.py` does both steps with NumPy alone, reading the
official files inside folders or zip archives (archive names change between
releases; the files are found by name). Chumpy arrays are replaced by NumPy
arrays with identical values, sparse matrices are stored densely, and the
MANO parameters are merged into the SMPL-H files, so loading needs neither
chumpy nor scipy. Files already cleaned with the official `smplx` tools work
as they are. When several versions of a file are present, the newest
official release is used and the others are listed; unreadable or invalid
files are reported and skipped while the other files are still converted;
the gender-neutral SMPL model (`basicmodel_neutral_...`) is ignored, this
pack offering the male and female models only. The output goes to
`ComfyUI/models/body_models/` when the pack is installed in
`ComfyUI/custom_nodes/`, and to the pack's own `models/` folder otherwise
(`--output` overrides both). Existing output files are
kept (`skip ... (exists; use --force to overwrite)`) and a run that only
finds them in place succeeds; `--force` replaces them, for example after
downloading a newer release.

## Shape regressor

`height_cm` and `weight_kg` are mapped to the first 10 SMPL shape
coefficients (betas) by a per-model, per-gender **linear regressor**:
`betas = [height_cm, weight_kg] · W + b`, with `W` a 2×10 matrix and `b` a
10-vector stored as plain NumPy arrays (`.npz`, loaded with
`allow_pickle=False`; no scikit-learn).

**How it is produced** (fully synthetic, deterministic —
[`smpl_avatar/regressor.py`](../smpl_avatar/regressor.py)):

1. sample 20 000 shape vectors `betas ~ N(0, I)` (seed 0);
2. build each rest-pose mesh `v_template + shapedirs · betas`;
3. measure its **height** (vertex extent along the vertical axis) and its
   **weight** (enclosed volume × 985 kg/m³). The volume of a closed mesh whose
   vertices are affine in the betas is a cubic polynomial in the betas; its
   coefficients are computed once from the mesh faces, which makes the whole
   fit take about 2 s per model;
4. fit ordinary least squares `[height, weight, 1] → betas` on 80 % of the
   samples;
5. report the **round-trip error**: rebuild the mesh from the predicted betas
   and compare its measured height and weight with the requested values.

**Why it is not shipped.** The coefficients are derived from the SMPL model,
whose license does not address derived data and forbids making the model
available "in any form" to third parties. Rather than distributing them under
MIT, the node computes them **on the user's machine from the user's own model
files** on first use (about 2 s) and caches them in
`models/regressor/height_weight_to_betas.npz`. It refits automatically if the
model file changes. To fit explicitly and print the report:

```bash
python scripts/train_regressor.py [--samples 20000] [--seed 0] [--report report.json]
```

(Run it with the Python that runs ComfyUI, since it loads the models through
`smplx`; Windows portable build:
`python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-SMPL\scripts\train_regressor.py ...`.)

**Fit error** (round trip, measured on SMPL v1.1.0 and MANO-merged SMPL+H
files; heights 155–200 cm for men and 145–190 cm for women):

| Model | Evaluation set | Height MAE | Weight MAE | Weight MAPE | Weight max |
|---|---|---|---|---|---|
| SMPL male | held-out bodies (4 000 samples of the SMPL prior) | 0.12 cm | 0.91 kg | 1.1 % | 18.5 kg |
| | height × BMI grid, BMI 18.5–30 | 0.10 cm | 1.07 kg | 1.6 % | 4.9 kg |
| | height × BMI grid, BMI 18–35 | 0.14 cm | 1.66 kg | 2.0 % | 12.7 kg |
| SMPL female | held-out bodies | 0.02 cm | 1.01 kg | 1.5 % | 21.7 kg |
| | grid, BMI 18.5–30 | 0.02 cm | 1.09 kg | 1.6 % | 6.7 kg |
| | grid, BMI 17–35 | 0.02 cm | 1.82 kg | 2.3 % | 15.2 kg |
| SMPL-H male | held-out bodies | 0.06 cm | 0.89 kg | 1.1 % | 22.0 kg |
| | grid, BMI 18.5–30 | 0.07 cm | 1.08 kg | 1.6 % | 4.9 kg |
| SMPL-H female | held-out bodies | < 0.01 cm | 1.02 kg | 1.5 % | 20.9 kg |
| | grid, BMI 18.5–30 | < 0.01 cm | 1.10 kg | 1.6 % | 6.7 kg |

Height is reproduced almost exactly. Weight is accurate for common body
shapes; the largest errors occur at extreme combinations (very tall and very
heavy, or very short and very light), because weight grows roughly with
height × girth², which a linear map cannot follow everywhere.

The node's sliders accept 120–220 cm and 30–200 kg, a wider range than the
table above was evaluated on (the fit itself covers roughly 150–215 cm and
35–150 kg for men, 135–195 cm and 25–140 kg for women). Beyond it the linear
map extrapolates and the error grows quickly: for example 220 cm / 165 kg
gives a mesh about 30 kg (men) or 50 kg (women) too heavy, and 120 cm /
200 kg is not a realistic body at all (|beta| above about 5). The node logs
the requested and achieved measurements for every render, and logs a
**warning** when the request is outside the fitted ranges, when the mesh
deviates by more than 1 cm or 5 % of the weight (minimum 2 kg), or when
|beta| exceeds 5. The deviation warning can also appear inside the evaluated
range at its short/light and tall/heavy corners (a 155 cm / 50 kg man comes
out about 3.5 kg heavy, for instance): it reports the achieved values, not
a failure.

"Height" is the vertical extent of the rest-pose mesh, from the top of the
head to the lowest point of the feet (the toe tips, about 1 cm below the
heels in the SMPL pose).

## Rendering

- **Camera**: `pyrender.OrthographicCamera`, looking at the body from the
  front (+z). The orthographic projection maps metres to pixels linearly with
  square pixels; `framing` controls the scale (see the node inputs).
- **Renderer**: one `pyrender.OffscreenRenderer` is kept alive and recreated
  only when the output size changes.
- **Render worker**: PyOpenGL binds to a single OpenGL platform per process.
  Recent ComfyUI versions import it at startup for their built-in GLSL node,
  bound to ANGLE (OpenGL ES only), which pyrender cannot use. When OpenGL was
  already imported by someone else, the OpenGL passes therefore run in a
  small persistent worker process (started on first use, restarted if it
  dies, stopped with ComfyUI) that exchanges plain NumPy arrays with the node;
  everything else runs in the ComfyUI process. Otherwise (older ComfyUI,
  scripts, tests) rendering happens in-process. This also keeps a graphics
  driver crash from taking ComfyUI down. A reply that does not arrive within
  `SMPL_AVATAR_RENDER_TIMEOUT` seconds (default 300) gets the worker killed
  and restarted, so a driver hang cannot freeze ComfyUI. The worker's own log
  lines are relayed to the ComfyUI log with a `render worker:` prefix. On
  laptops with an integrated and a discrete GPU, Windows may run the worker
  on the integrated one; rendering stays correct and the OpenGL renderer in
  use is logged once per process (`[SMPL Avatar] render worker: OpenGL
  renderer: ...` with the worker, `[SMPL Avatar] OpenGL renderer: ...`
  in-process).
- **In-process rendering** is tied to the thread that created the OpenGL
  context (ComfyUI runs nodes on one thread); other threads get a clear
  error and should use the worker.
- **Silhouette**: taken from the alpha channel of the render, i.e. the exact
  4× MSAA coverage of each pixel by the mesh — not from thresholding the
  coloured image. `soft_mask` returns the coverage itself; otherwise pixels
  with at least half of their samples on the body (alpha ≥ 127/255, see the
  next point) are 1. The shaded `image` is always composited with the exact
  coverage, so without `soft_mask` its anti-aliased edge extends about one
  pixel beyond the hard mask; multiply the image by the mask for a hard cut.
- **Masks and MSAA**: the hard mask keeps pixels with at least half of their
  4 samples on the body (alpha ≥ 127/255, which classifies exactly-half
  pixels identically on every driver). With `soft_mask`, the mask is the
  coverage itself and the depth map and part masks are multiplied by it, so
  an anti-aliased edge pixel's depth is its coverage times its depth value
  (it can drop below 0.2, the value of the farthest fully covered point).
- **Depth**: pyrender converts the depth buffer with the perspective formula
  even for orthographic cameras; the node inverts it to recover the true
  linear camera distance, then normalises it over the visible body. Only fully
  covered pixels are trusted (resolving multisampled depth is
  implementation-defined at silhouette edges); edge pixels take the mean of
  their fully covered neighbours.
- **Part masks**: a second, flat-shaded pass draws each part in its own
  colour. The colours are chosen so that no anti-aliased blend of two parts,
  or of a part and the background, can equal another part's colour. Pixels
  that match exactly are labelled directly, boundary pixels take the majority
  label of their neighbours, and each part mask is the label map times the
  silhouette mask, so the masks are disjoint and sum to `mask`.
- **Determinism**: masks, depth and part masks are bit-identical from one
  run to the next. The shaded image can differ by 1/255 on a few pixels
  between two identical renders on some GPUs and drivers; it comes from the
  GPU, not from the node.

### Part segmentation

The 24-part segmentation is the vertex segmentation published by Meshcapade
(`smpl_vert_segmentation.json`). Its repository states no license, so it is
**not bundled**: the node downloads it on first use from a pinned commit of
https://github.com/Meshcapade/wiki, checks its SHA-256 and stores it in the
pack's `models/segmentation/`. For offline machines run
`python scripts/download_segmentation.py` beforehand (with the Python that
runs ComfyUI; `--force` replaces a file that is not the pinned one), or copy
the file there manually, and set `SMPL_AVATAR_NO_DOWNLOAD=1`. Vertices on
part boundaries are
listed in several parts; each face gets the part that contains most of its
vertices (ties go to the part with more vertices exclusive to it, then to the
lower index). SMPL and SMPL-H share the same mesh topology, so the same
segmentation serves both.

## OpenGL platform and headless Linux

PyOpenGL selects its backend from the `PYOPENGL_PLATFORM` environment
variable, read once when OpenGL is first imported. This pack:

- **never overrides or removes an existing `PYOPENGL_PLATFORM`** — if you set
  it (`egl`, `osmesa`, ...), your choice is used as is;
- if it is unset **and** the process runs on Linux without a display
  (`DISPLAY` and `WAYLAND_DISPLAY` both unset), sets it to `egl` so rendering
  works on headless GPU servers;
- otherwise (Windows, macOS, Linux desktop) leaves it unset, and pyrender uses
  pyglet.

The decision is taken when ComfyUI imports the pack, before anything is
rendered, and logged. On recent ComfyUI versions that log line reads
`PYOPENGL_PLATFORM='egl' already set`: ComfyUI sets the variable for its own
GLSL node, and the render worker described below uses the native backend
instead.

When rendering runs in the worker process (see [Rendering](#rendering)), the
worker's environment is derived from ComfyUI's, which itself is never
modified:

- `PYOPENGL_PLATFORM` is passed on unchanged, **except** when its value is
  the `egl` that ComfyUI's GLSL node sets for ANGLE on Windows / macOS
  (detected by ComfyUI's `comfy_angle` module being loaded): that ANGLE
  setting cannot run pyrender, so the worker uses the platform default
  instead. On Linux, `egl` is kept.
- the headless-Linux rule above then applies to the worker as well.

Headless EGL needs the NVIDIA (or Mesa) EGL libraries, e.g. `libegl1` in
Docker images. `osmesa` (CPU rendering) should also work if OSMesa is
installed, with binary (not anti-aliased) masks because pyrender's OSMesa
backend returns no alpha channel; it has not been tested.

## The vendored pyrender

The renderer is [pyrender](https://github.com/mmatl/pyrender) (MIT), included
as a patched copy in [`smpl_avatar/_vendor`](../smpl_avatar/_vendor/README.md)
rather than installed from PyPI: the last PyPI release (0.1.45, 2021) pins
`PyOpenGL==3.1.0`, and installing it would **downgrade the PyOpenGL that
ComfyUI itself requires** (`>=3.1.8`) and break ComfyUI's built-in GLSL node.
The copy works with current PyOpenGL, NumPy 2 and recent Python versions; its
changes are listed in `smpl_avatar/_vendor/README.md` and the exact diff is in
`smpl_avatar/_vendor/pyrender.patch`.

`requirements.txt` asks for `PyOpenGL>=3.1.9`, which satisfies the
`>=3.1.8` that recent ComfyUI versions require. On an older ComfyUI whose
environment still has PyOpenGL 3.1.0 (pulled in by a PyPI pyrender), the pack
works as well, apart from the deprecation warnings 3.1.0 triggers on current
Python versions, so the upgrade is optional there — and reversible:
reinstalling such a pyrender simply puts 3.1.0 back.

## Compatibility

Supported Python versions: **3.10 to 3.14**, the versions ComfyUI runs on
(its Windows portable build ships Python 3.13). Python 3.15 (release
candidate) already passes the test suite.

Tested on 2026-09-29 with the dependencies a fresh `pip install -r
requirements.txt` resolves to (torch 2.14.0 CPU, NumPy 2.2.6 on 3.10, 2.4.6 on
3.11 and 2.5.3 on 3.12+, PyOpenGL 3.1.10, pyglet 2.1.16, trimesh 5.1.0, smplx
0.1.28):

| Python | Platform | Test suite (real SMPL / SMPL-H models) | Inside ComfyUI 0.37 |
|---|---|---|---|
| 3.10.21 | Windows 11 | all passed | not run |
| 3.11.16 | Windows 11 | all passed | not run |
| 3.12.14 | Windows 11 | all passed | workflow OK, PyOpenGL kept at 3.1.10, ComfyUI's GLSL node OK |
| 3.13.15 | Windows 11 | all passed | workflow OK, PyOpenGL kept at 3.1.10, ComfyUI's GLSL node OK |
| 3.14.7 | Windows 11 | all passed | workflow OK, PyOpenGL kept at 3.1.10, ComfyUI's GLSL node OK |
| 3.15.0rc2 | Windows 11 | all passed | not run |
| 3.10.12 | Linux, headless (WSL2 Ubuntu 22.04, Mesa EGL) | all passed | not run |

Also tested in ComfyUI 0.3.75 (Python 3.10, NumPy 1.26, PyOpenGL 3.1.0 and
3.1.10) and in the ComfyUI 0.41.0 Windows portable build (Python 3.13, torch
2.14, NumPy 2.5), including installation through ComfyUI-Manager.

In ComfyUI the node renders through the worker process (pyglet on Windows);
on headless Linux the pack selects EGL. Not tested yet: macOS, and a Linux GPU
server without WSL.

Notes:

- **Other node packs that install `pyrender` from PyPI.** pyrender 0.1.45
  pins `PyOpenGL==3.1.0`, which downgrades the PyOpenGL ComfyUI needs.
  Installing this pack's requirements brings PyOpenGL back to a version
  ComfyUI supports; `pip check` may then report the PyPI pyrender's pin. This
  pack does not use that package (it ships its own copy), so the message is
  harmless here; `pip uninstall pyrender` removes it if nothing else needs it.
- **Dependency bounds.** `pyglet` is capped below 3.0, whose pre-releases
  remove the `pyglet.gl` module the offscreen context needs. The PyOpenGL
  4.0 pre-release worked in a smoke test, so PyOpenGL is not capped.
- **Known third-party messages** (harmless): PyOpenGL's `No OpenGL_accelerate
  module loaded` line, printed at start-up by ComfyUI's own GLSL node on
  recent versions and again by the render worker on first render;
  DeprecationWarnings (hidden by
  default) from pyglet's use of `ctypes._pointer_type_cache` on Windows with
  Python 3.14+ and from PyOpenGL's use of `ctypes.__version__` on Python
  3.15; a `torch.jit` warning emitted by one of ComfyUI's own dependencies.
  smplx's "only 10 shape coefficients" message, printed when loading SMPL-H,
  is captured by the pack (10 betas is what it uses).
- **Memory**: `part_masks` is 24 × H × W float32, i.e. 96 bytes per pixel
  (96 MB at 1024×1024, 1.5 GB at 4096×4096); turn `compute_part_masks` off
  when you do not need it.
- No custom socket types: every input and output is a standard ComfyUI
  `IMAGE`, `MASK`, `INT`, `FLOAT`, `BOOLEAN` or combo.

## Development

Importing the pack is cheap: `torch`, `smplx` and the renderer are only
imported when a node runs.

```bash
pip install -r requirements.txt pytest
python -m pytest
```

The tests run without any licensed file. They cover the regressor (fitting,
exact volume polynomial, `.npz` I/O), mask union and label decoding,
segmentation parsing, the OpenGL platform policy, camera framing, node
registration, and an end-to-end run of both nodes on **synthetic
SMPL-format models** (an ellipsoid with the SMPL mesh sizes, written in the
official file format including fake chumpy objects), plus guards on the
dependency list. Rendering tests skip only when the machine cannot create an
OpenGL context (any other rendering error fails them; set
`SMPL_AVATAR_REQUIRE_GL=1` to never skip). Tests on the real models run when
they are installed (for example with `SMPL_AVATAR_MODEL_DIR=...`) and skip
otherwise.
