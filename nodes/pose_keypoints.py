"""SMPL Pose Keypoints: the OpenPose skeleton of the views of Render SMPL Views."""

from __future__ import annotations

import json

from ..smpl_avatar import keypoints
from .common import CATEGORY, VIEWS_TYPE
from .tensors import stack_views, to_tensor


class SMPLAvatarPoseKeypoints:
    DESCRIPTION = (
        "OpenPose skeleton (COCO 18 keypoints) of the body in the views of 'Render SMPL Views', "
        "aligned with its renders: an image for OpenPose ControlNets, the keypoints in the "
        "POSE_KEYPOINT format of comfyui_controlnet_aux, and every body joint with the cameras as JSON."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "views": (VIEWS_TYPE, {"tooltip": "The 'views' output of 'Render SMPL Views'."}),
            },
            "optional": {
                "include_face": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add the nose, eyes and ears where they are seen in the render (a back "
                               "view has none, as an OpenPose detector would report it).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_image", "keypoints", "joints_json")
    OUTPUT_TOOLTIPS = (
        "OpenPose skeleton on black, one per view or as a sheet, like the renders.",
        "Keypoints in pixels, OpenPose JSON layout (POSE_KEYPOINT of comfyui_controlnet_aux).",
        "JSON: per view, the camera (OpenCV convention), the COCO keypoints and every SMPL body joint.",
    )
    FUNCTION = "keypoints"
    CATEGORY = CATEGORY

    def keypoints(self, views, include_face=True):
        view_set = views
        per_view = keypoints.keypoints_per_view(view_set, include_face)
        images = [keypoints.draw(k, view_set.width, view_set.height) for k in per_view]
        frames = keypoints.pose_frames(view_set, per_view)
        report = keypoints.joints_report(view_set, per_view)
        return (to_tensor(stack_views(images, view_set.layout)), frames, json.dumps(report, indent=1))
