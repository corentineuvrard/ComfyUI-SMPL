"""Lazy SMPL / SMPL-H loading through ``smplx``, cached per (model_type, gender)."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import paths
from .regressor import NUM_BETAS, ShapeMeasurer, model_fingerprint

LOGGER = logging.getLogger(__name__)

MODEL_TYPES = ("SMPL", "SMPL-H")
GENDERS = ("male", "female")

# SMPL / SMPL-H kinematic tree: joint 16 = left shoulder, 17 = right shoulder.
# body_pose excludes the root joint, hence the -1.
_LEFT_SHOULDER_POSE = (16 - 1) * 3
_RIGHT_SHOULDER_POSE = (17 - 1) * 3


class BodyModelFormatError(RuntimeError):
    """The model file exists but is not in the format smplx expects."""


@dataclass
class BodyModel:
    model_type: str
    gender: str
    path: Path
    module: Any  # smplx.SMPL or smplx.SMPLH
    faces: np.ndarray  # (F, 3) int64
    v_template: np.ndarray  # (V, 3) float32
    shapedirs: np.ndarray  # (V, 3, NUM_BETAS) float32
    _fingerprint: Optional[str] = field(default=None, repr=False)
    _measurer: Optional[ShapeMeasurer] = field(default=None, repr=False)
    face_labels: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = model_fingerprint(self.v_template, self.shapedirs, self.faces)
        return self._fingerprint

    @property
    def measurer(self) -> ShapeMeasurer:
        """Rest-pose height/weight of shapes of this model (see regressor.ShapeMeasurer)."""
        if self._measurer is None:
            self._measurer = ShapeMeasurer(self.v_template, self.shapedirs, self.faces)
        return self._measurer


_lock = threading.Lock()
_cache: Dict[Tuple[str, str], BodyModel] = {}


def _import_smplx():
    try:
        import torch
        import smplx
    except ImportError as exc:
        raise ImportError(
            "The 'smplx' package is required (pip install -r requirements.txt). Note that smplx "
            "is distributed under the Max Planck non-commercial research license."
        ) from exc
    return torch, smplx


def _validate(model_type: str, gender: str) -> None:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    if gender not in GENDERS:
        raise ValueError(f"gender must be one of {GENDERS}, got {gender!r}")


def _build(model_type: str, gender: str, path: Path) -> BodyModel:
    torch, smplx = _import_smplx()
    common = dict(model_path=str(path), gender=gender, num_betas=NUM_BETAS, batch_size=1)
    try:
        if model_type == "SMPL":
            module = smplx.SMPL(**common)
        else:
            module = smplx.SMPLH(
                **common, ext=path.suffix.lstrip("."), use_pca=False, flat_hand_mean=False
            )
    except ModuleNotFoundError as exc:
        if "chumpy" in str(exc):
            raise BodyModelFormatError(
                f"{path} still contains chumpy objects, which smplx cannot load without the "
                "unmaintained chumpy package. Convert it first with "
                "'python scripts/prepare_models.py' (see README, 'Body model files')."
            ) from exc
        raise
    except (AttributeError, KeyError) as exc:
        if model_type == "SMPL-H" and "hands_" in str(exc):
            raise BodyModelFormatError(
                f"{path} has no MANO hand parameters. SMPL-H files from the MANO website must be "
                "merged with MANO_LEFT/MANO_RIGHT first (see README, 'Body model files')."
            ) from exc
        raise
    module.eval()
    return BodyModel(
        model_type=model_type,
        gender=gender,
        path=path,
        module=module,
        faces=np.asarray(module.faces, dtype=np.int64),
        v_template=module.v_template.detach().cpu().numpy().astype(np.float32),
        shapedirs=module.shapedirs.detach().cpu().numpy()[:, :, :NUM_BETAS].astype(np.float32),
    )


def load(model_type: str, gender: str) -> BodyModel:
    """Return the cached model, loading it on first use."""
    _validate(model_type, gender)
    key = (model_type, gender)
    with _lock:
        if key not in _cache:
            path = paths.find_body_model(model_type, gender)
            LOGGER.info("[SMPL Avatar] Loading %s %s from %s", model_type, gender, path)
            _cache[key] = _build(model_type, gender, path)
        return _cache[key]


def cached_keys():
    with _lock:
        return sorted(_cache)


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def posed_vertices(body: BodyModel, betas: np.ndarray, arm_angle_deg: float = 0.0):
    """Vertices ``(V, 3)`` and joints ``(J, 3)`` in metres, y-up, facing +z.

    ``arm_angle_deg`` lowers both arms from the T-pose by rotating the shoulder
    joints about the viewing axis (0 = T-pose, about 45 = A-pose).
    """
    torch, _ = _import_smplx()
    module = body.module
    betas_t = torch.as_tensor(np.asarray(betas, dtype=np.float32).reshape(1, -1))
    body_pose = torch.zeros(1, module.NUM_BODY_JOINTS * 3)
    angle = math.radians(arm_angle_deg)
    body_pose[0, _LEFT_SHOULDER_POSE + 2] = -angle
    body_pose[0, _RIGHT_SHOULDER_POSE + 2] = angle
    with torch.no_grad():
        output = module(betas=betas_t, body_pose=body_pose, return_verts=True)
    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = output.joints[0].detach().cpu().numpy().astype(np.float32)
    return vertices, joints
