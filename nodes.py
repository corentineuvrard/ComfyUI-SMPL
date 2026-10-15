"""ComfyUI node definitions (classic V1 node API)."""

from __future__ import annotations

import numpy as np

from .smpl_avatar import body_model, masks, pipeline, render
from .smpl_avatar.segmentation import NUM_PARTS, PART_DESCRIPTIONS, PART_NAMES

CATEGORY = "SMPL Avatar"


def _to_tensor(array: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def grow_mask(mask, pixels: int):
    """Dilate (``pixels > 0``) or erode (``pixels < 0``) a ``(N, H, W)`` mask with a square kernel."""
    if not pixels:
        return mask
    import torch.nn.functional as F

    radius = abs(int(pixels))
    x = mask[:, None]
    if pixels > 0:
        y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    else:
        # max_pool2d pads with -inf; after negation that would count everything
        # outside the image as inside the mask, so pad with zeros explicitly.
        x = F.pad(x, (radius, radius, radius, radius), value=0.0)
        y = -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1)
    return y[:, 0]


class SMPLAvatarGenerate:
    DESCRIPTION = (
        "Build an SMPL or SMPL-H body whose height and weight match the inputs, and render it "
        "with an orthographic camera. Outputs a shaded image, the silhouette mask computed from "
        "the exact render coverage, a depth map (near = white, background = black) and 24 "
        "per-part masks for 'Union SMPL Part Masks'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list(body_model.MODEL_TYPES), {
                    "default": "SMPL",
                    "tooltip": "SMPL (body only) or SMPL-H (body with articulated hands). "
                               "The model files must be installed; see the README.",
                }),
                "gender": (list(body_model.GENDERS), {"default": "male"}),
                "height_cm": ("FLOAT", {
                    "default": 175.0, "min": 120.0, "max": 220.0, "step": 0.5,
                    "tooltip": "Target body height in centimetres (standing, vertical extent of the "
                               "body). Fitted on about 150-215 cm (male) / 135-195 cm (female); "
                               "outside that range, or when the mesh deviates from the request, "
                               "the achieved height and weight are logged with a warning.",
                }),
                "weight_kg": ("FLOAT", {
                    "default": 70.0, "min": 30.0, "max": 200.0, "step": 0.5,
                    "tooltip": "Target body weight in kilograms, obtained from the mesh volume "
                               "at 985 kg/m^3. Most accurate for BMI 17-35 within the height "
                               "range above; a warning is logged when the mesh deviates.",
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                                  "tooltip": "Output image width in pixels."}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8,
                                   "tooltip": "Output image height in pixels. part_masks costs 96 bytes "
                                              "per pixel (1.5 GB at 4096x4096)."}),
            },
            "optional": {
                "arm_angle": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 75.0, "step": 1.0,
                    "tooltip": "Lower both arms from the T-pose by this many degrees (about 45 = A-pose).",
                }),
                "yaw": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Rotate the body about the vertical axis, in degrees (0 = facing the "
                               "camera, 90 = right profile: the body's right side faces the camera).",
                }),
                "framing": (list(render.FRAMINGS), {
                    "default": "fit",
                    "tooltip": "fit: scale the body to fill the image. fixed: constant metres per "
                               "pixel (view_height_m tall), so different heights stay comparable.",
                }),
                "padding": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "fit: extra view size around the body, as a fraction of the body "
                               "extent (0.1: the body spans 1/1.1 of the limiting dimension). "
                               "fixed: gap below the feet, as a fraction of half the view height.",
                }),
                "view_height_m": ("FLOAT", {
                    "default": 2.2, "min": 0.5, "max": 5.0, "step": 0.05,
                    "tooltip": "Height of the orthographic view in metres (framing = fixed only).",
                }),
                "background": (list(render.BACKGROUNDS), {"default": "black"}),
                "soft_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: binary masks (at least half of the pixel covered). On: "
                               "anti-aliased masks equal to the exact per-pixel coverage; the depth "
                               "map and part masks are scaled by that coverage too (edge pixels are "
                               "no longer held at or above the 0.2 depth floor).",
                }),
                "compute_part_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Render the 24 body-part masks (24 x H x W float32: 96 MB at "
                               "1024x1024). Needs the SMPL part segmentation, downloaded on first "
                               "use. When off, part_masks is all zeros.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask", "depth", "part_masks")
    OUTPUT_TOOLTIPS = (
        "Shaded render composited over the chosen background.",
        "Silhouette mask (1 = body).",
        "Depth map: nearest point white, farthest fully covered body point 0.2 gray, background "
        "black (with soft_mask, edge pixels are scaled by their coverage).",
        "24 masks, one per SMPL body part in canonical order (hips first); all zeros when "
        "compute_part_masks is off. Use 'Union SMPL Part Masks'.",
    )
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, model_type, gender, height_cm, weight_kg, width, height,
                 arm_angle=0.0, yaw=0.0, framing="fit", padding=0.1, view_height_m=2.2,
                 background="black", soft_mask=False, compute_part_masks=True):
        avatar = pipeline.generate(
            model_type, gender, height_cm, weight_kg, width, height,
            arm_angle=arm_angle, yaw=yaw, framing=framing, padding=padding,
            view_height_m=view_height_m, background=background, soft_mask=soft_mask,
            part_masks=compute_part_masks,
        )
        result = avatar.render
        part_masks = result.part_masks
        if part_masks is None:
            part_masks = np.zeros((NUM_PARTS, height, width), dtype=np.float32)
        depth_rgb = np.repeat(result.depth[..., None], 3, axis=2)
        return (
            _to_tensor(result.image)[None],
            _to_tensor(result.mask)[None],
            _to_tensor(depth_rgb)[None],
            _to_tensor(part_masks),
        )


class SMPLAvatarPartMaskUnion:
    DESCRIPTION = (
        "Combine SMPL body-part masks into a single mask. Select parts with a preset, the "
        "individual toggles, or both (the selection is the union of the two)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "part_masks": ("MASK", {"tooltip": "The 'part_masks' output of 'Generate SMPL Avatar'."}),
            "preset": (list(masks.PRESETS), {"default": "none", "tooltip": "Group of parts to include."}),
        }
        for name in PART_NAMES:
            required[name] = ("BOOLEAN", {"default": False, "tooltip": PART_DESCRIPTIONS[name]})
        return {
            "required": required,
            "optional": {
                "grow_px": ("INT", {
                    "default": 0, "min": -64, "max": 64, "step": 1,
                    "tooltip": "Dilate (positive) or erode (negative) the result by this many pixels.",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Union of the selected parts (1 = selected).",)
    FUNCTION = "union"
    CATEGORY = CATEGORY

    def union(self, part_masks, preset="none", grow_px=0, **toggles):
        names = set(masks.PRESETS[preset])
        names.update(name for name in PART_NAMES if toggles.get(name))
        array = part_masks.detach().cpu().numpy() if hasattr(part_masks, "detach") else np.asarray(part_masks)
        result = masks.union_part_masks(array, masks.part_indices(names))
        return (grow_mask(_to_tensor(result), grow_px),)


NODE_CLASS_MAPPINGS = {
    "SMPLAvatarGenerate": SMPLAvatarGenerate,
    "SMPLAvatarPartMaskUnion": SMPLAvatarPartMaskUnion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLAvatarGenerate": "Generate SMPL Avatar",
    "SMPLAvatarPartMaskUnion": "Union SMPL Part Masks",
}
