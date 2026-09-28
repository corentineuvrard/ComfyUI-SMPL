# Third-party notices

The source code in this repository is released under the MIT license (see
`LICENSE`). It **does not include** any of the third-party data or code listed
below. Users obtain these separately, under their own terms, which are **not**
MIT and in several cases **do not allow commercial use**.

## Data that users download themselves

### SMPL body model
- Owner: Max-Planck-Gesellschaft (Max Planck Institute for Intelligent Systems).
- Source: https://smpl.is.tue.mpg.de (registration and license acceptance required).
- License: SMPL-Model License, for non-commercial scientific research,
  non-commercial education and non-commercial artistic projects. Redistribution
  is not allowed. Commercial licenses are offered through Meshcapade GmbH.
- Not included in this repository. The shape regressor used by the nodes is
  computed on each user's machine from their own copy of the model and is not
  distributed either.
- Please cite: M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, M. J. Black,
  "SMPL: A Skinned Multi-Person Linear Model", ACM Transactions on Graphics
  (Proc. SIGGRAPH Asia), 2015.

### SMPL+H (SMPL-H) and MANO
- Owner: Max-Planck-Gesellschaft.
- Source: https://mano.is.tue.mpg.de (registration and license acceptance required).
- License: MANO license, non-commercial; redistribution not allowed.
- Not included in this repository.
- Please cite: J. Romero, D. Tzionas, M. J. Black, "Embodied Hands: Modeling
  and Capturing Hands and Bodies Together", ACM Transactions on Graphics
  (Proc. SIGGRAPH Asia), 2017.

### SMPL 24-part vertex segmentation (`smpl_vert_segmentation.json`)
- Published by Meshcapade GmbH in its body-modelling wiki repository:
  https://github.com/Meshcapade/wiki, file
  `assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json`
  (commit `be6a56e7a84f85339c0801da50b3ac028833c807`, 2021-06-08).
- License: the repository does not state a license for this file.
- Not included in this repository. The node pack downloads it on first use
  from the pinned commit above and verifies its SHA-256
  (`0af78b5c581a225998b4804b4c9dd8351e155b34253947eb2541160852105ab8`).
  Set `SMPL_AVATAR_NO_DOWNLOAD=1` to disable the download.

## Python packages installed as dependencies

| Package | License | How it is used |
|---|---|---|
| [smplx](https://github.com/vchoutas/smplx) | "Software Copyright License for non-commercial scientific research purposes" (Max Planck; SMPL-X / SMPLify-X terms). Not an open-source license. | Loads SMPL / SMPL-H models at runtime. Installed from PyPI by the user; no smplx code is copied into this repository. |
| [pyrender](https://github.com/mmatl/pyrender) | MIT | Offscreen OpenGL rendering. |
| [trimesh](https://github.com/mikedh/trimesh) | MIT | Dependency of pyrender. |
| [PyOpenGL](https://github.com/mcfletch/pyopengl) | BSD-style | Dependency of pyrender. |
| [pyglet](https://github.com/pyglet/pyglet) | BSD 3-Clause | Dependency of pyrender (windowing on Windows / macOS / desktop Linux). |
| NumPy, PyTorch | BSD 3-Clause | Provided by ComfyUI. |

`scripts/prepare_models.py` reads model files that contain
[chumpy](https://github.com/mattloper/chumpy) (MIT) objects **without**
importing chumpy; no chumpy code is used or copied.

This node pack runs inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
(GPL-3.0) but contains no ComfyUI code.

"SMPL", "SMPL+H" and "MANO" are used to describe compatibility with these
models; they are the names of models owned by the Max-Planck-Gesellschaft. This
project is not affiliated with or endorsed by the Max Planck Institute or
Meshcapade.
