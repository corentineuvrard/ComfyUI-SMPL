"""Union SMPL Part Masks: combine body-part masks into a single mask."""

from __future__ import annotations

import logging

import numpy as np

from ..smpl_avatar import masks
from ..smpl_avatar.segmentation import PART_DESCRIPTIONS, PART_NAMES
from .common import CATEGORY
from .tensors import grow_mask, to_tensor

LOGGER = logging.getLogger(__name__)


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
        if not array.any():
            LOGGER.warning(
                "[SMPL Avatar] Union SMPL Part Masks: every input part mask is empty, so the "
                "output is empty too: usually 'compute_part_masks' is off on Generate SMPL "
                "Avatar, or the body lies outside the view (framing = fixed)."
            )
        return (grow_mask(to_tensor(result), grow_px),)
