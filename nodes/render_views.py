"""Render SMPL Views: render a body from one or several views, as a batch or a sheet."""

from __future__ import annotations

from ..smpl_avatar import pipeline
from ..smpl_avatar import views as smpl_views
from ..smpl_avatar.segmentation import NUM_PARTS
from .common import AVATAR_TYPE, CATEGORY, VIEWS_TYPE
from .generate import SMPLAvatarGenerate
from .tensors import to_tensor


class SMPLAvatarRenderViews:
    DESCRIPTION = (
        "Render a body from one or several views (front, back, profiles, three-quarter views, "
        "turnarounds, or any azimuth and elevation), with an orthographic or perspective camera. "
        "All views share one scale. 'batch' outputs one image per view; 'sheet' puts the views "
        "side by side in a single image (a character sheet), with every map aligned."
    )

    @classmethod
    def INPUT_TYPES(cls):
        generate = SMPLAvatarGenerate.INPUT_TYPES()
        return {
            "required": {
                "avatar": (AVATAR_TYPE, {"tooltip": "The output of 'SMPL Avatar Body'."}),
                "views": (list(smpl_views.VIEW_PRESETS), {
                    "default": "turnaround 4",
                    "tooltip": "Views to render, in this order. Left and right are the character's own "
                               "sides. 'custom' is a single view at the given azimuth and elevation.",
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                                  "tooltip": "Width of each view in pixels."}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8,
                                   "tooltip": "Height of each view in pixels. The part masks take 96 bytes "
                                              "per pixel and view (about 300 MB for 8 views at 512 x 768)."}),
            },
            "optional": {
                "azimuth": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Degrees. The view angle for 'custom'; with the other views, turns every "
                               "view by this angle. 0 = front, 90 = right profile, 180 = back.",
                }),
                "elevation": ("FLOAT", {
                    "default": 0.0, "min": -85.0, "max": 85.0, "step": 1.0,
                    "tooltip": "Degrees, for every view: positive looks down at the body from above, "
                               "negative looks up at it from below.",
                }),
                "projection": (list(smpl_views.PROJECTIONS), {
                    "default": "orthographic",
                    "tooltip": "orthographic: no perspective, sizes do not depend on depth (character "
                               "sheets, multi-view models). perspective: a regular camera, like a photo.",
                }),
                "fov": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Vertical field of view in degrees (perspective only). Smaller = longer "
                               "lens and flatter look; the camera moves so that the body keeps its framing.",
                }),
                "framing": (generate["optional"]["framing"][0], {
                    **generate["optional"]["framing"][1],
                    "tooltip": "fit: scale the body to fill the images (one scale for all views). fixed: "
                               "set the view height with view_height_m.",
                }),
                "padding": generate["optional"]["padding"],
                "view_height_m": ("FLOAT", {
                    "default": 2.2, "min": 0.5, "max": 5.0, "step": 0.05,
                    "tooltip": "Height of the view in metres for framing = fixed (measured at the "
                               "body for perspective). A warning is logged when the body is cut.",
                }),
                "layout": (list(pipeline.LAYOUTS), {
                    "default": "sheet",
                    "tooltip": "batch: one image per view. sheet: all views side by side in one image.",
                }),
                "background": generate["optional"]["background"],
                "soft_mask": generate["optional"]["soft_mask"],
                "compute_part_masks": (generate["optional"]["compute_part_masks"][0], {
                    **generate["optional"]["compute_part_masks"][1],
                    "tooltip": "Render the 24 body-part masks of every view. Needs the SMPL part "
                               "segmentation, downloaded on first use. When off, part_masks is all zeros.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", VIEWS_TYPE)
    RETURN_NAMES = ("image", "mask", "depth", "part_masks", "views")
    OUTPUT_TOOLTIPS = (
        "Shaded renders: one per view, or the sheet.",
        "Silhouette masks (1 = body), aligned with the images.",
        "Depth maps (near = white, background = black), each view normalised on its own.",
        "24 part masks per view or per sheet (24 x N masks for N images), for 'Union SMPL Part Masks'.",
        "The views and cameras used, for 'SMPL Pose Keypoints'.",
    )
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, avatar, views, width, height, azimuth=0.0, elevation=0.0, projection="orthographic",
               fov=30.0, framing="fit", padding=0.1, view_height_m=2.2, layout="sheet",
               background="black", soft_mask=False, compute_part_masks=True):
        import torch

        view_set = pipeline.prepare_views(
            avatar, views, width, height, azimuth=azimuth, elevation=elevation, projection=projection,
            fov=fov, framing=framing, padding=padding, view_height_m=view_height_m, layout=layout,
        )
        count = len(view_set.specs)
        # Outputs are allocated once and filled view by view, so that only one view's
        # intermediate arrays are alive at a time. 24 part masks per output image, as
        # 'Union SMPL Part Masks' expects: (24 N, H, W) for a batch, (24, H, N W) for a sheet.
        if layout == "sheet":
            images, masks_out, depths = (torch.zeros(1, height, count * width, 3), torch.zeros(1, height, count * width),
                                         torch.zeros(1, height, count * width, 3))
            parts = torch.zeros(NUM_PARTS, height, count * width)
        else:
            images, masks_out, depths = (torch.zeros(count, height, width, 3), torch.zeros(count, height, width),
                                         torch.zeros(count, height, width, 3))
            parts = torch.zeros(NUM_PARTS * count, height, width)
        for i in range(count):
            result = pipeline.render_view(view_set, i, background=background, soft_mask=soft_mask,
                                          part_masks=compute_part_masks)
            if layout == "sheet":
                cols = slice(i * width, (i + 1) * width)
                targets = (images[0, :, cols], masks_out[0, :, cols], depths[0, :, cols], parts[:, :, cols])
            else:
                targets = (images[i], masks_out[i], depths[i], parts[NUM_PARTS * i:NUM_PARTS * (i + 1)])
            targets[0].copy_(to_tensor(result.image))
            targets[1].copy_(to_tensor(result.mask))
            targets[2].copy_(to_tensor(result.depth)[..., None].expand(height, width, 3))
            if result.part_masks is not None:
                targets[3].copy_(to_tensor(result.part_masks))
            del result
        return (images, masks_out, depths, parts, view_set)
