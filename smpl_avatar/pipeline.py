"""Height/weight -> betas -> posed mesh -> rendered maps."""

from __future__ import annotations

import logging
import math
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


HEIGHT_TOLERANCE_CM = 1.0
WEIGHT_TOLERANCE_KG = 2.0
WEIGHT_TOLERANCE_FRACTION = 0.05
BETA_LIMIT = 5.0  # |beta| beyond this gives unrealistic bodies


def extrapolation_warning(
    reg: regressor.LinearBetaRegressor,
    height_cm: float,
    weight_kg: float,
    measured_h: float,
    measured_w: float,
    betas: np.ndarray,
) -> Optional[str]:
    """Explain when a request lies outside what the linear regressor reproduces well.

    Returns ``None`` when the request is inside the fitted ranges and the mesh
    matches it, otherwise a one-line message with the achieved values.
    """
    # Training bounds are the min/max of random samples: round them outward so
    # that a request at the stated bound is not reported as outside it.
    h_range = reg.meta.get("training_height_range_cm")
    w_range = reg.meta.get("training_weight_range_kg")
    outside = []
    if h_range and not (math.floor(h_range[0]) <= height_cm <= math.ceil(h_range[1])):
        outside.append(f"height {height_cm:.0f} cm is outside the fitted range "
                       f"{math.floor(h_range[0])}-{math.ceil(h_range[1])} cm")
    if w_range and not (math.floor(w_range[0]) <= weight_kg <= math.ceil(w_range[1])):
        outside.append(f"weight {weight_kg:.0f} kg is outside the fitted range "
                       f"{math.floor(w_range[0])}-{math.ceil(w_range[1])} kg")
    deviates = []
    weight_tolerance = max(WEIGHT_TOLERANCE_KG, WEIGHT_TOLERANCE_FRACTION * weight_kg)
    if abs(measured_h - height_cm) > HEIGHT_TOLERANCE_CM or abs(measured_w - weight_kg) > weight_tolerance:
        deviates.append(f"the mesh measures {measured_h:.1f} cm / {measured_w:.1f} kg")
    beta_max = float(np.abs(np.asarray(betas)).max())
    if beta_max > BETA_LIMIT:
        deviates.append(f"|beta| reaches {beta_max:.1f} (above {BETA_LIMIT:.0f} the body shape becomes unrealistic)")
    if not outside and not deviates:
        return None
    if outside:
        return ("the request is beyond the range the height/weight regressor was fitted on: "
                + "; ".join(outside + deviates))
    return "the linear height/weight regressor does not match this request exactly: " + "; ".join(deviates)


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
    warning = extrapolation_warning(reg, height_cm, weight_kg, float(measured_h[0]), float(measured_w[0]), betas)
    if warning:
        LOGGER.warning("[SMPL Avatar] %s %s: %s", model_type, gender, warning)
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
