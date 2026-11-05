"""ComfyUI node definitions (classic V1 node API), one module per node."""

from .generate import SMPLAvatarGenerate
from .part_masks import SMPLAvatarPartMaskUnion

NODE_CLASS_MAPPINGS = {
    "SMPLAvatarGenerate": SMPLAvatarGenerate,
    "SMPLAvatarPartMaskUnion": SMPLAvatarPartMaskUnion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLAvatarGenerate": "Generate SMPL Avatar",
    "SMPLAvatarPartMaskUnion": "Union SMPL Part Masks",
}
