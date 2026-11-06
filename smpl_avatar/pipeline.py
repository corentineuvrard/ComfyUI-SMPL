"""Height/weight -> betas -> posed mesh -> rendered maps (one view or a set of views)."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import body_model, keypoints, paths, regressor, segmentation, views
from .render import RenderResult, View, compute_view, render_mesh, rotate_y

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


def clipping_warning(vertices: np.ndarray, view) -> Optional[str]:
    """Describe how the body overflows the orthographic ``view``, or ``None`` if it fits.

    ``fit`` framing always fits. ``fixed`` framing keeps the view
    ``view_height_m`` metres tall and ``view_height_m * width / height`` metres
    wide, whatever the body does.
    """
    v = np.asarray(vertices, dtype=np.float64)
    (xmin, ymin), (xmax, ymax) = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    left, right = view.center_x - view.xmag, view.center_x + view.xmag
    bottom, top = view.center_y - view.ymag, view.center_y + view.ymag
    tolerance = 1e-6
    cuts = []
    if xmin < left - tolerance:
        cuts.append(f"{left - xmin:.2f} m on the left")
    if xmax > right + tolerance:
        cuts.append(f"{xmax - right:.2f} m on the right")
    if ymax > top + tolerance:
        cuts.append(f"{ymax - top:.2f} m at the top")
    if ymin < bottom - tolerance:
        cuts.append(f"{bottom - ymin:.2f} m at the bottom")
    if not cuts:
        return None
    return (
        f"the body ({xmax - xmin:.2f} m wide, {ymax - ymin:.2f} m tall) does not fit the "
        f"{2 * view.xmag:.2f} x {2 * view.ymag:.2f} m view and is cut by " + ", ".join(cuts)
        + ". With framing = fixed the view is view_height_m x width/height metres wide: "
        "increase view_height_m, use a wider image, or lower the arms with arm_angle."
    )


def _check_finite(*named_values) -> None:
    for name, value in named_values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")


@dataclass
class AvatarBody:
    """A posed body in its own frame (metres, y up, facing +z), before any view."""

    model: body_model.BodyModel
    model_type: str
    gender: str
    height_cm: float
    weight_kg: float
    arm_angle: float
    betas: np.ndarray
    vertices: np.ndarray  # (V, 3)
    joints: np.ndarray  # (J, 3): the model's joints, then smplx's face, feet and fingertip points
    measured_height_cm: float
    measured_weight_kg: float

    @property
    def faces(self) -> np.ndarray:
        return self.model.faces


def build_body(
    model_type: str,
    gender: str,
    height_cm: float,
    weight_kg: float,
    arm_angle: float = 0.0,
    regressor_path: Optional[Path] = None,
) -> AvatarBody:
    """Load the model, map height and weight to betas and pose the arms."""
    _check_finite(("height_cm", height_cm), ("weight_kg", weight_kg), ("arm_angle", arm_angle))
    model = body_model.load(model_type, gender)
    reg = regressor_for(model, regressor_path or paths.REGRESSOR_PATH)
    betas = reg.predict(height_cm, weight_kg)
    measured_h, measured_w = model.measurer(betas)
    LOGGER.info(
        "[SMPL Avatar] %s %s: requested %.1f cm / %.1f kg -> mesh %.1f cm / %.1f kg",
        model_type, gender, height_cm, weight_kg, measured_h[0], measured_w[0],
    )
    warning = extrapolation_warning(reg, height_cm, weight_kg, float(measured_h[0]), float(measured_w[0]), betas)
    if warning:
        LOGGER.warning("[SMPL Avatar] %s %s: %s", model_type, gender, warning)
    vertices, joints = body_model.posed_vertices(model, betas, arm_angle)
    return AvatarBody(
        model=model, model_type=model_type, gender=gender, height_cm=float(height_cm),
        weight_kg=float(weight_kg), arm_angle=float(arm_angle),
        betas=np.asarray(betas, dtype=np.float32), vertices=vertices, joints=joints,
        measured_height_cm=float(measured_h[0]), measured_weight_kg=float(measured_w[0]),
    )


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
    # Widget ranges are checked by ComfyUI; values from linked inputs are not.
    _check_finite(("height_cm", height_cm), ("weight_kg", weight_kg), ("arm_angle", arm_angle),
                  ("yaw", yaw), ("padding", padding), ("view_height_m", view_height_m))
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding!r}")
    if view_height_m <= 0:
        raise ValueError(f"view_height_m must be positive, got {view_height_m!r}")
    body = build_body(model_type, gender, height_cm, weight_kg, arm_angle, regressor_path)
    vertices = rotate_y(body.vertices, yaw)
    joints = rotate_y(body.joints, yaw)
    view = compute_view(vertices, width, height, framing, padding, view_height_m)
    clipped = clipping_warning(vertices, view)
    if clipped:
        LOGGER.warning("[SMPL Avatar] %s %s: %s", model_type, gender, clipped)
    labels = face_labels_for(body.model) if part_masks else None
    result = render_mesh(
        vertices, body.faces, width, height, view,
        face_labels=labels, background=background, soft_mask=soft_mask,
    )
    return Avatar(
        render=result,
        betas=body.betas,
        vertices=vertices,
        joints=joints,
        measured_height_cm=body.measured_height_cm,
        measured_weight_kg=body.measured_weight_kg,
    )


# ------------------------------------------------------------- multi-view


@dataclass
class ViewSet:
    """The views rendered by 'Render SMPL Views', passed on to 'SMPL Pose Keypoints'."""

    body: AvatarBody
    specs: List[views.ViewSpec]
    cameras: List[View]
    pivot: np.ndarray
    width: int
    height: int
    layout: str  # "batch" or "sheet"
    # Per view: which face keypoints (keypoints.COCO_FROM_VERTEX names) the view
    # shows (facing the camera, inside the image and not hidden in the rendered
    # depth); filled by render_view(), {} for a view not rendered yet.
    face_visible: List[Dict[str, bool]] = field(default_factory=list)

    def posed(self, points: np.ndarray, index: int) -> np.ndarray:
        """``points`` of the body (rest frame) as seen in view ``index``."""
        return views.transform(points, self.specs[index], self.pivot)


LAYOUTS = ("batch", "sheet")


def prepare_views(
    body: AvatarBody,
    preset: str,
    width: int,
    height: int,
    azimuth: float = 0.0,
    elevation: float = 0.0,
    projection: str = "orthographic",
    fov: float = 30.0,
    framing: str = "fit",
    padding: float = 0.1,
    view_height_m: float = 2.2,
    layout: str = "batch",
) -> ViewSet:
    """The views of ``preset`` and their cameras, all at the same scale (nothing rendered yet)."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    specs = views.expand(preset, azimuth, elevation)
    pivot = views.pivot_of(body.vertices)
    posed = [views.transform(body.vertices, spec, pivot) for spec in specs]
    cameras = views.frame_views(posed, width, height, projection, fov, framing, padding, view_height_m)
    cut = [spec.label for spec, verts, cam in zip(specs, posed, cameras)
           if views.outside_fraction(verts, cam, width, height) > 0]
    if cut:
        LOGGER.warning(
            "[SMPL Avatar] %s %s: the body does not fit the view in %s. With framing = fixed, increase "
            "view_height_m, use a wider image, or lower the arms with arm_angle.",
            body.model_type, body.gender, ", ".join(cut),
        )
    return ViewSet(body=body, specs=specs, cameras=cameras, pivot=pivot,
                   width=int(width), height=int(height), layout=layout)


def render_view(
    view_set: ViewSet,
    index: int,
    background: str = "black",
    soft_mask: bool = False,
    part_masks: bool = True,
) -> RenderResult:
    """Render view ``index`` of ``view_set`` and record which face keypoints it shows."""
    body = view_set.body
    labels = face_labels_for(body.model) if part_masks else None
    result = render_mesh(view_set.posed(body.vertices, index), body.faces, view_set.width, view_set.height,
                         view_set.cameras[index], face_labels=labels, background=background, soft_mask=soft_mask)
    visible = keypoints.face_visibility(view_set, index, result.distance, result.coverage)
    while len(view_set.face_visible) <= index:  # views may be rendered in any order
        view_set.face_visible.append({})
    view_set.face_visible[index] = visible
    return result
