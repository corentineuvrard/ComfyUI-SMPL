# Third-party notices

The source code in this repository is released under the MIT license (see
`LICENSE`), except the vendored copy of pyrender in
`smpl_avatar/_vendor/pyrender`, which keeps its own MIT license (see "Code
included in this repository" below). The repository **does not include** the
third-party data or Python packages listed below: users obtain them
separately, under their own terms, which are **not** MIT and in several cases
**do not allow commercial use**.

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

## Demo images in this repository

`docs/demo_outputs.png` and `docs/demo_part_unions.png` are 2D renders
(shaded image, silhouette, 8-bit depth, part labels; 320 × 480 pixels per
tile) produced by this node pack from the official SMPL and SMPL+H models;
`docs/workflow_maps.png` and `docs/workflow_part_masks.png` are screenshots
of the example workflows running in ComfyUI and contain renders of the same
kind (rendered at 512 × 768, shown scaled down). They contain no model data
(no mesh, vertices, blend shapes or weights). They
are shown for illustration and education only, are **not** covered by this
repository's MIT license, and remain subject to the SMPL-Model and MANO
licenses of the models they were rendered from. SMPL and SMPL+H body models
© Max-Planck-Gesellschaft; see `docs/README.md` for the citations.

## Code included in this repository

### pyrender (vendored)
- Source: https://github.com/mmatl/pyrender, commit
  `a59963ef890891656fd17c90e12d663233dcaa99` (2022-04-30).
- License: MIT, "Copyright (c) 2019 Matthew Matl"; the full license text is in
  `smpl_avatar/_vendor/pyrender/LICENSE` and applies to the files in that
  folder.
- Modified: viewer and text rendering removed, imports made relative, NumPy 2
  and Python 3 clean-ups, one error-path fix. See
  `smpl_avatar/_vendor/README.md` and the exact diff in
  `smpl_avatar/_vendor/pyrender.patch`.

## Python packages installed as dependencies

| Package | License | How it is used |
|---|---|---|
| [smplx](https://github.com/vchoutas/smplx) | "Software Copyright License for non-commercial scientific research purposes" (Max Planck; SMPL-X / SMPLify-X terms). Not an open-source license. | Loads SMPL / SMPL-H models at runtime. Installed from PyPI by the user; no smplx code is copied into this repository. |
| [trimesh](https://github.com/mikedh/trimesh) | MIT | Used by the vendored pyrender. |
| [networkx](https://github.com/networkx/networkx) | BSD 3-Clause | Scene graph of the vendored pyrender. |
| [PyOpenGL](https://github.com/mcfletch/pyopengl) | BSD-style | OpenGL bindings used by the vendored pyrender. |
| [pyglet](https://github.com/pyglet/pyglet) | BSD 3-Clause | OpenGL context creation on Windows / macOS / desktop Linux. |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU (HPND) | Image utilities of the vendored pyrender. |
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
