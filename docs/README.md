# docs/

Illustrations for the main README.

## Demo images and the SMPL license

`demo_outputs.png` and `demo_part_unions.png` are renders produced by the
*Generate SMPL Avatar* and *Union SMPL Part Masks* nodes from the official
SMPL and SMPL+H body models, which are owned by the Max-Planck-Gesellschaft
and distributed under the non-commercial SMPL-Model / MANO licenses
(https://smpl.is.tue.mpg.de, https://mano.is.tue.mpg.de).
`workflow_maps.png` and `workflow_part_masks.png` are screenshots of the two
workflows of `example_workflows/` after running them in ComfyUI; the previews
they show are renders of the same kind, and the same terms apply to them.

- They are 2D results (shaded render, silhouette, 8-bit depth, part labels)
  at 320 × 480 pixels per tile; the previews in the workflow screenshots were
  rendered at 512 × 768 and appear scaled down (about 420 × 630 pixels). They
  contain no mesh, vertex, blend-shape or skinning data, and the model files
  themselves are not part of this repository.
- They are shown for illustration and education only. They are **not**
  covered by the MIT license of this repository; the Max Planck licenses
  apply to the models they were rendered from.
- Credit: SMPL and SMPL+H body models © Max-Planck-Gesellschaft. Please cite
  M. Loper et al., "SMPL: A Skinned Multi-Person Linear Model" (SIGGRAPH Asia
  2015) and, for SMPL+H, J. Romero et al., "Embodied Hands" (SIGGRAPH Asia
  2017). This project is not affiliated with or endorsed by the Max Planck
  Institute for Intelligent Systems or Meshcapade GmbH.
- The 24-part colouring uses the body-part segmentation published by
  Meshcapade GmbH (see `THIRD_PARTY_NOTICES.md`); the colours are arbitrary.

How they were made: the bodies of the contact sheets were rendered with the
node's default lighting at 320 × 480, `fit` framing, from the heights,
weights, arm angles and yaws written in each row's caption; the "mesh ... cm /
... kg" values are the node's own measurements of the generated mesh. The
workflow screenshots were taken in ComfyUI after running the example workflows
unchanged.
