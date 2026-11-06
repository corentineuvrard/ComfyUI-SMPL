"""ComfyUI node definitions (classic V1 node API), one module per node."""

from .body import SMPLAvatarBody
from .generate import SMPLAvatarGenerate
from .part_masks import SMPLAvatarPartMaskUnion
from .pose_keypoints import SMPLAvatarPoseKeypoints
from .render_views import SMPLAvatarRenderViews

NODE_CLASS_MAPPINGS = {
    "SMPLAvatarGenerate": SMPLAvatarGenerate,
    "SMPLAvatarPartMaskUnion": SMPLAvatarPartMaskUnion,
    "SMPLAvatarBody": SMPLAvatarBody,
    "SMPLAvatarRenderViews": SMPLAvatarRenderViews,
    "SMPLAvatarPoseKeypoints": SMPLAvatarPoseKeypoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMPLAvatarGenerate": "Generate SMPL Avatar",
    "SMPLAvatarPartMaskUnion": "Union SMPL Part Masks",
    "SMPLAvatarBody": "SMPL Avatar Body",
    "SMPLAvatarRenderViews": "Render SMPL Views",
    "SMPLAvatarPoseKeypoints": "SMPL Pose Keypoints",
}
