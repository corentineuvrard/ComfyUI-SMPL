# ComfyUI-SMPL — SMPL Avatar nodes

ComfyUI nodes that build an **SMPL** or **SMPL-H** body from a **height and a
weight**, render it with an **orthographic camera**, and output maps that can
be used directly as ControlNet-style conditioning:

- a shaded render (`IMAGE`),
- the silhouette as a proper `MASK`, computed from the exact render coverage,
- a depth map (`IMAGE`, near = white),
- 24 per-body-part masks (`MASK`), plus a node that unions any selection of parts.

> **Licensing in one paragraph.** The code in this repository is MIT-licensed.
> It depends on the SMPL / SMPL-H body models and on the `smplx` package, which
> are **not** MIT: both are distributed by the Max Planck Institute for
> Intelligent Systems under **non-commercial research licenses**, and the model
> files cannot be redistributed. Every user must download the models from the
> official websites and accept their license. Commercial use of SMPL requires a
> license from Meshcapade / Max Planck. See [Licensing](#licensing) and
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Nodes

### Generate SMPL Avatar (`SMPLAvatarGenerate`)

| Input | Type | Description |
|---|---|---|
| `model_type` | `SMPL` / `SMPL-H` | Body only, or body with articulated hands (relaxed hand pose). |
| `gender` | `male` / `female` | Selects the gendered model. |
| `height_cm`, `weight_kg` | FLOAT | Target body measurements (see [Shape regressor](#shape-regressor)). |
| `width`, `height` | INT | Output size in pixels. |
| `arm_angle` *(optional)* | FLOAT | Lowers both arms from the T-pose (0 = T-pose, ~45 = A-pose). |
| `yaw` *(optional)* | FLOAT | Rotation about the vertical axis in degrees (90 = right profile). |
| `framing` *(optional)* | `fit` / `fixed` | `fit` fills the image; `fixed` keeps a constant metres-per-pixel scale so different heights stay comparable. |
| `padding` *(optional)* | FLOAT | Margin around the body (`fit`) or below the feet (`fixed`). |
| `view_height_m` *(optional)* | FLOAT | Height of the view in metres for `fixed` framing. |
| `background` *(optional)* | `black` / `white` / `gray` | Background of the shaded render. |
| `soft_mask` *(optional)* | BOOLEAN | Anti-aliased masks (exact per-pixel coverage) instead of binary ones. |
| `compute_part_masks` *(optional)* | BOOLEAN | Render the 24 part masks (needs the part segmentation, downloaded on first use). |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE `[1,H,W,3]` | Shaded render composited over the background. |
| `mask` | MASK `[1,H,W]` | Silhouette, 1 = body. |
| `depth` | IMAGE `[1,H,W,3]` | Nearest point 1.0, farthest visible body point 0.2, background 0. |
| `part_masks` | MASK `[24,H,W]` | One mask per SMPL part, in the order of the table below. The 24 masks are disjoint and their sum equals `mask`. |

### Union SMPL Part Masks (`SMPLAvatarPartMaskUnion`)

Takes `part_masks` and returns one `MASK` `[1,H,W]`: the union of the parts
selected by a `preset` (head, torso, arms, hands, legs, feet, upper body,
lower body, full body, ...) and/or the 24 individual toggles. `grow_px` dilates
(positive) or erodes (negative) the result. A batch of `24·N` masks from `N`
avatars gives `N` output masks.

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

## Installation

1. Clone this repository into `ComfyUI/custom_nodes/` (or install it with
   ComfyUI-Manager).
2. Install the requirements in the Python environment that runs ComfyUI:

   ```bash
   pip install -r requirements.txt
   ```

   This installs `smplx` and the rendering dependencies (`PyOpenGL`,
   `pyglet`, `trimesh`, `networkx`, `Pillow`). `torch` and `numpy` come with
   ComfyUI and are deliberately
   not listed; `opencv` and `scikit-learn` are not needed.
3. Install the body model files (next section) and restart ComfyUI.

The nodes appear under the **SMPL Avatar** category. Importing the pack is
cheap: `torch`, `smplx` and the renderer are only imported when a node runs.

The renderer is [pyrender](https://github.com/mmatl/pyrender) (MIT), included
as a patched copy in [`smpl_avatar/_vendor`](smpl_avatar/_vendor/README.md)
rather than installed from PyPI: the last PyPI release (0.1.45, 2021) pins
`PyOpenGL==3.1.0`, and installing it would **downgrade the PyOpenGL that
ComfyUI itself requires** (`>=3.1.8`) and break ComfyUI's built-in GLSL node.
The copy works with current PyOpenGL, NumPy 2 and recent Python versions; its
changes are listed in `smpl_avatar/_vendor/README.md` and the exact diff is in
`smpl_avatar/_vendor/pyrender.patch`.
An example workflow (render, silhouette, depth and an upper-body part mask) is
available from ComfyUI's template browser, or in
[`example_workflows/smpl_avatar_maps.json`](example_workflows/smpl_avatar_maps.json).

## Body model files

The SMPL and SMPL-H files are **not** included and must never be committed to
this repository (they are git-ignored). Each user downloads them:

| Model | Where | What to download |
|---|---|---|
| SMPL | https://smpl.is.tue.mpg.de | The "SMPL for Python" archive (e.g. `SMPL_python_v.1.1.0.zip`), which contains the male and female models `basicmodel_{m,f}_lbs_10_207_0_v1.x.0.pkl` |
| SMPL-H | https://mano.is.tue.mpg.de | The "Models & Code" archive (e.g. `mano_v1_2.zip`), which contains `SMPLH_male.pkl`, `SMPLH_female.pkl`, `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` |

Both sites require registration and acceptance of a non-commercial license.
Archive names may change between releases; the preparation script below finds
the model files by name inside any folder or zip archive.

### Preparing the files

The official files cannot be loaded by `smplx` as downloaded:

1. **They contain `chumpy` objects.** The `smplx` documentation requires
   removing them before use (its `tools/clean_ch.py` must run in a
   *Python 2* environment with chumpy installed; chumpy does not work with
   current Python / NumPy releases).
2. **SMPL-H must be merged with the MANO hand parameters.** `smplx` reads the
   hand PCA basis and mean pose (`hands_componentsl/r`, `hands_meanl/r`) from
   the SMPL-H file itself; its `tools/merge_smplh_mano.py` does that merge
   (also a Python 2 script).

This repository ships a Python 3 script that does both steps **without
chumpy**. Point it at the folder where you saved the downloads (zip archives
are read directly, no need to extract them):

```bash
python scripts/prepare_models.py --source path/to/downloads
```

It writes the layout expected by `smplx`:

```
models/
├── smpl/SMPL_MALE.pkl, SMPL_FEMALE.pkl
└── smplh/SMPLH_MALE.pkl, SMPLH_FEMALE.pkl
```

Chumpy arrays are replaced by NumPy arrays with identical values and sparse
matrices are stored densely, so loading needs neither chumpy nor scipy. Files
already cleaned with the official `smplx` tools work as they are.

### Where the files are searched

1. every folder listed in the environment variable `SMPL_AVATAR_MODEL_DIR`
   (separated by `;` on Windows, `:` elsewhere),
2. `ComfyUI/models/body_models/`,
3. this pack's `models/` folder.

Each location uses the `smpl/` + `smplh/` layout above (folder and file names
are matched case-insensitively; `smpl-h/` is accepted too). Models are loaded
lazily on first use and cached per (model type, gender).

## Shape regressor

`height_cm` and `weight_kg` are mapped to the first 10 SMPL shape
coefficients (betas) by a per-model, per-gender **linear regressor**:
`betas = [height_cm, weight_kg] · W + b`, with `W` a 2×10 matrix and `b` a
10-vector stored as plain NumPy arrays (`.npz`, loaded with
`allow_pickle=False`; no scikit-learn).

**How it is produced** (fully synthetic, deterministic —
[`smpl_avatar/regressor.py`](smpl_avatar/regressor.py)):

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
height × girth², which a linear map cannot follow everywhere. The node logs
the requested and achieved measurements for every render.

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
  driver crash from taking ComfyUI down.
- **Silhouette**: taken from the alpha channel of the render, i.e. the exact
  4× MSAA coverage of each pixel by the mesh — not from thresholding the
  coloured image. `soft_mask` returns the coverage itself; otherwise pixels
  with coverage ≥ 50 % are 1.
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

### Part segmentation

The 24-part segmentation is the vertex segmentation published by Meshcapade
(`smpl_vert_segmentation.json`). Its repository states no license, so it is
**not bundled**: the node downloads it on first use from a pinned commit of
https://github.com/Meshcapade/wiki, checks its SHA-256 and stores it in
`models/segmentation/`. For offline machines run
`python scripts/download_segmentation.py` beforehand (or copy the file there
manually) and set `SMPL_AVATAR_NO_DOWNLOAD=1`. Vertices on part boundaries are
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
rendered, and logged.

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
Docker images. `osmesa` (CPU) works too if OSMesa is installed.

### Environment variables

| Variable | Effect |
|---|---|
| `SMPL_AVATAR_MODEL_DIR` | Extra folder(s) searched for body model files. |
| `SMPL_AVATAR_NO_DOWNLOAD=1` | Never download the part segmentation. |
| `SMPL_AVATAR_RENDER_PROCESS` | `auto` (default), `worker` or `inprocess`. |
| `SMPL_AVATAR_GL_PLATFORM` | `PYOPENGL_PLATFORM` for the render worker (`egl`, `osmesa`, ...; `default` = unset). |
| `PYOPENGL_PLATFORM` | Respected as described above. |

## Compatibility

Tested on 2026-09-28:

- **Windows 11, inside ComfyUI 0.37.0** (frontend 1.53.6): both nodes in an API
  workflow and in the example workflow run from the editor, chained with core
  nodes (`MaskToImage`, `GrowMask`, `SaveImage`, `PreviewImage`); rendering
  in the worker process with pyglet.
- **Windows 11, test suite**: Python 3.11 / NumPy 1.26 and Python 3.10 /
  NumPy 2.2, PyTorch 2.9, smplx 0.1.28, pyrender 0.1.45, pyglet 2.1.
- **Linux, headless**: WSL2 Ubuntu 22.04 with `DISPLAY` unset, Mesa EGL,
  Python 3.10, NumPy 2.2, PyTorch 2.14 (CPU): the pack selects EGL and the full
  test suite passes. Not yet tested on a physical Linux GPU server or on macOS.

Notes:

- No custom socket types: every input and output is a standard ComfyUI
  `IMAGE`, `MASK`, `INT`, `FLOAT`, `BOOLEAN` or combo.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest
```

The tests run without any licensed file. They cover the regressor (fitting,
exact volume polynomial, `.npz` I/O), mask union and label decoding,
segmentation parsing, the OpenGL platform policy, camera framing, node
registration, and an end-to-end run of both nodes on **synthetic
SMPL-format models** (an ellipsoid with the SMPL mesh sizes, written in the
official file format including fake chumpy objects). Rendering tests skip
when no OpenGL context is available. Tests on the real models run when they
are installed (for example with `SMPL_AVATAR_MODEL_DIR=...`) and skip
otherwise.

## Licensing

- **This repository's code**: MIT, © 2026 Corentin Euvrard ([LICENSE](LICENSE)).
- **smplx** (runtime dependency, installed from PyPI): Max Planck
  "Software Copyright License for non-commercial scientific research
  purposes". It is not an open-source license, and using this pack means
  using smplx under those terms.
- **SMPL, SMPL-H, MANO model files**: Max Planck non-commercial licenses; each
  user downloads them and accepts the license; redistribution is not allowed.
  Commercial licensing of SMPL is available through Meshcapade.
- **Part segmentation**: published by Meshcapade without a stated license;
  downloaded by each user, not redistributed.
- **Shape regressor coefficients**: derived from the user's own SMPL files on
  the user's machine; not distributed.

In short, the MIT license applies to this code only; it does not make the
SMPL stack as a whole usable for commercial purposes. Please cite SMPL
(Loper et al., 2015) and, for SMPL-H, MANO (Romero et al., 2017) in
publications — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
