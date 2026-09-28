"""Height/weight -> betas -> posed mesh -> rendered maps."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import body_model, paths, regressor, segmentation
from .render import RenderResult, compute_view, render_mesh, rotate_y

LOGGER = logging.getLogger(__name__)

_regressor_lock = threading.Lock()
_regressors: Dict[str, regressor.LinearBetaRegressor] = {}


def regressor_for(body: body_model.BodyModel, path: Path = paths.REGRESSOR_PATH) -> regressor.LinearBetaRegressor:
    """Return the regressor for ``body``, training and caching it if needed.

    The regressor is generated locally from the user's own model file (it is
    not shipped with the node pack) and stored in ``path``. It is retrained
    automatically if the stored one was fitted on a different model file.
    """
    key = regressor.regressor_key(body.model_type, body.gender)
    path = Path(path)
    with _regressor_lock:
        cached = _regressors.get(key)
        if cached is not None and cached.meta.get("model_fingerprint") == body.fingerprint:
            return cached
        stored = {}
        if path.is_file():
            try:
                stored = regressor.load_regressors(path)
            except Exception as exc:
                LOGGER.warning("[SMPL Avatar] Ignoring unreadable regressor file %s: %s", path, exc)
        reg = stored.get(key)
        if reg is None or reg.meta.get("model_fingerprint") != body.fingerprint:
            LOGGER.info(
                "[SMPL Avatar] Fitting the height/weight regressor for %s %s (one-off)...",
                body.model_type, body.gender,
            )
            start = time.perf_counter()
            reg = regressor.train(
                body.v_template, body.shapedirs, body.faces, gender=body.gender,
                extra_meta={"model_type": body.model_type, "model_file": body.path.name},
            )
            report = reg.meta["report"]["held_out"]
            LOGGER.info(
                "[SMPL Avatar] Regressor ready in %.1f s (round-trip MAE: %.2f cm, %.2f kg)",
                time.perf_counter() - start, report["height_mae_cm"], report["weight_mae_kg"],
            )
            stored[key] = reg
            try:
                regressor.save_regressors(path, stored)
            except OSError as exc:
                LOGGER.warning("[SMPL Avatar] Could not save the regressor to %s: %s", path, exc)
        _regressors[key] = reg
        return reg


def face_labels_for(body: body_model.BodyModel) -> np.ndarray:
    if body.face_labels is None:
        membership = segmentation.load_membership()
        if body.faces.max() >= membership.shape[0]:
            raise ValueError("the SMPL part segmentation does not match this mesh topology")
        body.face_labels = segmentation.face_part_labels(membership, body.faces)
    return body.face_labels


@dataclass
class Avatar:
    render: RenderResult
    betas: np.ndarray
    vertices: np.ndarray
    joints: np.ndarray
    measured_height_cm: float
    measured_weight_kg: float


def generate(
    model_type: str,
    gender: str,
    height_cm: float,
    weight_kg: float,
    width: int,
    height: int,
    arm_angle: float = 0.0,
    yaw: float = 0.0,
    framing: str = "fit",
    padding: float = 0.1,
    view_height_m: float = 2.2,
    background: str = "black",
    soft_mask: bool = False,
    part_masks: bool = True,
    regressor_path: Optional[Path] = None,
) -> Avatar:
    body = body_model.load(model_type, gender)
    reg = regressor_for(body, regressor_path or paths.REGRESSOR_PATH)
    betas = reg.predict(height_cm, weight_kg)
    measured_h, measured_w = body.measurer(betas)
    LOGGER.info(
        "[SMPL Avatar] %s %s: requested %.1f cm / %.1f kg -> mesh %.1f cm / %.1f kg",
        model_type, gender, height_cm, weight_kg, measured_h[0], measured_w[0],
    )
    vertices, joints = body_model.posed_vertices(body, betas, arm_angle)
    vertices = rotate_y(vertices, yaw)
    joints = rotate_y(joints, yaw)
    view = compute_view(vertices, width, height, framing, padding, view_height_m)
    labels = face_labels_for(body) if part_masks else None
    result = render_mesh(
        vertices, body.faces, width, height, view,
        face_labels=labels, background=background, soft_mask=soft_mask,
    )
    return Avatar(
        render=result,
        betas=np.asarray(betas, dtype=np.float32),
        vertices=vertices,
        joints=joints,
        measured_height_cm=float(measured_h[0]),
        measured_weight_kg=float(measured_w[0]),
    )
