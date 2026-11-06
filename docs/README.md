# docs/

Illustrations for the main README.

## Demo images and the SMPL license

`demo_outputs.gif` (an animation) and `demo_part_unions.png` are renders
produced by this node pack from the official SMPL and SMPL+H body models,
which are owned by the Max-Planck-Gesellschaft
and distributed under the non-commercial SMPL-Model / MANO licenses
(https://smpl.is.tue.mpg.de, https://mano.is.tue.mpg.de).
`workflow_maps.png`, `workflow_part_masks.png` and
`workflow_character_sheet.png` are screenshots of the workflows of
`example_workflows/` after running them in ComfyUI; the previews they show
are renders of the same kind, and the same terms apply to them, as they do to
the template thumbnails `example_workflows/*.jpg`, which are cropped from
these screenshots.

- They are 2D results (shaded render, silhouette, 8-bit depth, part labels,
  OpenPose skeleton) at 160 × 240 pixels per panel in the animation and
  320 × 480 per tile in `demo_part_unions.png`; the previews in the workflow
  screenshots were rendered at 512 × 768 (384 × 576 per view in
  `workflow_character_sheet.png`) and appear scaled down. They contain no
  mesh, vertex, blend-shape or skinning data, and the model files themselves
  are not part of this repository.
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

How they were made: the animation shows SMPL+H bodies with the arms lowered
by 45°, rendered with the default lighting at 160 × 240 per panel and `fixed`
framing (a 2.12 m tall view, so that the sizes compare) while they turn by 5°
per frame; on each turn the camera goes from eye level in the front view to
30° above at the right profile, back to eye level in the back view and 30°
below at the left profile. Starting from 175 cm and 77 kg (a BMI of 25), a
male body grows to 195 cm and back, a female body shrinks to 155 cm and back,
then gains weight up to 95 kg and back, and a male body loses weight down to
58 kg and back; the same sequence follows with height and weight swapped (he
gains weight, she loses weight and then grows, he shrinks). The bodies swap in
the front view at 175 cm and 77 kg. The caption gives the requested values.
`demo_part_unions.png` was rendered with the default lighting at 320 × 480
with `fit` framing. The workflow screenshots were taken in ComfyUI after
running the example workflows unchanged.
