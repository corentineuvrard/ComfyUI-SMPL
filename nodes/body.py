"""SMPL Avatar Body: build a body without rendering it, for Render SMPL Views."""

from __future__ import annotations

from ..smpl_avatar import pipeline
from .common import AVATAR_TYPE, CATEGORY
from .generate import SMPLAvatarGenerate


def _body_inputs():
    """Inputs shared by the nodes that build a body."""
    generate = SMPLAvatarGenerate.INPUT_TYPES()
    required = {name: generate["required"][name] for name in ("model_type", "gender", "height_cm", "weight_kg")}
    return required, {"arm_angle": generate["optional"]["arm_angle"]}


class SMPLAvatarBody:
    DESCRIPTION = (
        "Build an SMPL or SMPL-H body whose height and weight match the inputs, without rendering "
        "it. Connect it to 'Render SMPL Views' to render it from several views."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required, optional = _body_inputs()
        return {"required": required, "optional": optional}

    RETURN_TYPES = (AVATAR_TYPE,)
    RETURN_NAMES = ("avatar",)
    OUTPUT_TOOLTIPS = ("The posed body, for 'Render SMPL Views'.",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, model_type, gender, height_cm, weight_kg, arm_angle=0.0):
        return (pipeline.build_body(model_type, gender, height_cm, weight_kg, arm_angle),)
