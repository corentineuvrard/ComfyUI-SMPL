"""2D keypoints of a body in each view, in the OpenPose body (COCO 18) layout.

Body keypoints come from the model's joints (the usual SMPL to OpenPose
correspondence: neck = SMPL neck joint, shoulders, elbows, wrists, hips, knees,
ankles). The nose, eyes and ears are not joints: they are mesh vertices, whose
indices smplx publishes (``smplx.vertex_ids``). Whether a face keypoint is
seen depends on where the head faces: the head's forward axis (from between
the ears to the nose) and side axis (from ear to ear) give each feature a
direction, compared with the direction to the camera, so that a profile seen
at eye level shows the nose, the near eye and the near ear, and a back view no
face, as an OpenPose detector would report them. A feature that another part
of the body hides in the rendered depth (an arm in front of the face) is
dropped too, as is any keypoint that falls outside the image.

The drawing follows the OpenPose / ControlNet convention: black background,
one colour per limb at 60 % intensity, keypoints as full-colour dots.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from . import views
from .render import OrthoView

COCO_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
    "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)
# COCO keypoint -> SMPL joint index (the first 22 joints are shared by SMPL and SMPL-H).
COCO_FROM_JOINT = {1: 12, 2: 17, 3: 19, 4: 21, 5: 16, 6: 18, 7: 20, 8: 2, 9: 5, 10: 8, 11: 1, 12: 4, 13: 7}
# COCO keypoint -> name of the vertex in smplx.vertex_ids["smplh"].
COCO_FROM_VERTEX = {0: "nose", 14: "reye", 15: "leye", 16: "rear", 17: "lear"}

# Limbs (pairs of COCO indices) and colours, in the OpenPose drawing order.
LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)
COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
)

SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)
NUM_BODY_JOINTS = 22  # shared by SMPL and SMPL-H; SMPL adds left_hand / right_hand
# Direction of each face feature in the head frame (forward f, right side r)
# and the minimum cosine between it and the direction to the camera for the
# feature to count as seen.
FEATURE_DIRECTIONS = {"nose": (1.0, 0.0), "reye": (1.0, 0.5), "leye": (1.0, -0.5), "rear": (0.3, 1.0), "lear": (0.3, -1.0)}
FEATURE_THRESHOLDS = {"nose": -0.3, "reye": 0.1, "leye": 0.1, "rear": 0.0, "lear": 0.0}
# A feature is hidden by another part of the body when the rendered surface
# around its pixel is at least this much closer to the camera (metres); the
# margin ignores the face's own relief (eye sockets, the ear's back).
OCCLUSION_MARGIN_M = 0.08


def face_vertex_ids() -> dict:
    """Vertex indices of the nose, eyes and ears (SMPL topology), as published by smplx."""
    from smplx.vertex_ids import vertex_ids

    return {name: int(vertex_ids["smplh"][name]) for name in COCO_FROM_VERTEX.values()}


def joint_names(model_type: str) -> List[str]:
    """Names of the body joints reported in the JSON output."""
    return list(SMPL_JOINT_NAMES if model_type == "SMPL" else SMPL_JOINT_NAMES[:NUM_BODY_JOINTS])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def face_visibility(view_set, index: int, distance: np.ndarray, coverage: np.ndarray,
                    face_ids: Optional[dict] = None) -> dict:
    """Which face features of view ``index`` are seen (by ``smplx.vertex_ids`` name).

    ``distance`` is the depth of the rendered surface along the camera axis,
    per pixel (0 = background), and ``coverage`` the pixel coverage, both
    from the render of that view; they only serve to detect occlusion by
    another body part.
    """
    face_ids = face_ids if face_ids is not None else face_vertex_ids()
    camera, width, height = view_set.cameras[index], view_set.width, view_set.height
    names = list(COCO_FROM_VERTEX.values())
    points = dict(zip(names, view_set.posed(view_set.body.vertices[[face_ids[n] for n in names]], index)))
    forward = _unit(points["nose"] - (points["rear"] + points["lear"]) / 2.0)
    side = points["rear"] - points["lear"]
    side = _unit(side - (side @ forward) * forward)
    uv = camera.project(np.array([points[n] for n in names]), width, height)
    result = {}
    for name, (u, v) in zip(names, uv):
        point = points[name]
        if isinstance(camera, OrthoView):
            to_camera = np.array([0.0, 0.0, 1.0])
        else:
            to_camera = _unit(np.array([camera.center_x, camera.center_y, camera.camera_z]) - point)
        a, b = FEATURE_DIRECTIONS[name]
        facing = float(_unit(a * forward + b * side) @ to_camera) > FEATURE_THRESHOLDS[name]
        depth = camera.camera_z - point[2]
        inside = bool(np.isfinite(u) and np.isfinite(v) and 0 <= u < width and 0 <= v < height and depth > 0)
        occluded = False
        if inside:
            x, y = int(u), int(v)
            window = (slice(max(y - 1, 0), y + 2), slice(max(x - 1, 0), x + 2))
            surface = distance[window][(coverage[window] > 0) & (distance[window] > 0)]
            occluded = bool(surface.size) and float(np.median(surface)) < depth - OCCLUSION_MARGIN_M
        result[name] = facing and inside and not occluded
    return result


def coco_keypoints(view_set, index: int, include_face: bool = True,
                   face_ids: Optional[dict] = None) -> np.ndarray:
    """``(18, 3)`` array of ``(x, y, confidence)`` in pixels for view ``index`` (confidence 0 = absent)."""
    body, camera = view_set.body, view_set.cameras[index]
    width, height = view_set.width, view_set.height
    points = np.zeros((len(COCO_NAMES), 3), dtype=np.float64)
    present = np.zeros(len(COCO_NAMES), dtype=bool)

    joints = view_set.posed(body.joints[:NUM_BODY_JOINTS], index)
    for coco, joint in COCO_FROM_JOINT.items():
        points[coco] = joints[joint]
        present[coco] = True

    if include_face:
        face_ids = face_ids if face_ids is not None else face_vertex_ids()
        visible = view_set.face_visible[index] if index < len(view_set.face_visible) else {}
        posed_vertices = view_set.posed(body.vertices[[face_ids[n] for n in COCO_FROM_VERTEX.values()]], index)
        for (coco, name), position in zip(COCO_FROM_VERTEX.items(), posed_vertices):
            points[coco] = position
            present[coco] = visible.get(name, False)

    uv = camera.project(points, width, height)
    finite = np.isfinite(uv).all(axis=1)
    inside = finite & (np.nan_to_num(uv[:, 0], nan=-1) >= 0) & (np.nan_to_num(uv[:, 0], nan=-1) < width) \
        & (np.nan_to_num(uv[:, 1], nan=-1) >= 0) & (np.nan_to_num(uv[:, 1], nan=-1) < height)
    keep = present & inside
    result = np.zeros((len(COCO_NAMES), 3), dtype=np.float64)
    result[keep, :2] = uv[keep]
    result[keep, 2] = 1.0
    return result


def body_joints_2d(view_set, index: int) -> List[dict]:
    """Every body joint of view ``index``: name, pixel position and depth along the camera axis."""
    body, camera = view_set.body, view_set.cameras[index]
    names = joint_names(body.model_type)
    joints = view_set.posed(body.joints[:len(names)], index)
    uv = camera.project(joints, view_set.width, view_set.height)
    depth = camera.camera_z - joints[:, 2]

    def number(value, digits):  # JSON has no NaN: a point behind the camera has no pixel
        return round(float(value), digits) if np.isfinite(value) else None

    return [
        {"name": name, "x": number(u, 3), "y": number(v, 3), "depth_m": number(d, 5)}
        for name, (u, v), d in zip(names, uv, depth)
    ]


def _ellipse(center, half_length: float, half_width: float, angle: float, steps: int = 36) -> List[tuple]:
    t = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    x = half_length * np.cos(t)
    y = half_width * np.sin(t)
    c, s = np.cos(angle), np.sin(angle)
    return [(float(center[0] + c * xi - s * yi), float(center[1] + s * xi + c * yi)) for xi, yi in zip(x, y)]


def draw(keypoints: np.ndarray, width: int, height: int, scale: Optional[float] = None) -> np.ndarray:
    """OpenPose-style skeleton ``(H, W, 3)`` float32 in [0, 1] for one ``(18, 3)`` keypoint array.

    Limb half-width and dot radius are 4 pixels at 512 pixels (the size
    OpenPose ControlNets were trained at) and grow with the image.
    """
    scale = scale if scale is not None else max(1.0, min(width, height) / 512.0)
    stick, radius = 4.0 * scale, 4.0 * scale
    canvas = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
    pen = ImageDraw.Draw(canvas)
    k = np.asarray(keypoints, dtype=np.float64)
    for (a, b), color in zip(LIMBS, COLORS):
        if k[a, 2] <= 0 or k[b, 2] <= 0:
            continue
        p, q = k[a, :2], k[b, :2]
        length = float(np.hypot(*(q - p)))
        angle = float(np.arctan2(q[1] - p[1], q[0] - p[0]))
        pen.polygon(_ellipse((p + q) / 2.0, length / 2.0, stick, angle), fill=tuple(int(c * 0.6) for c in color))
    for (x, y, confidence), color in zip(k, COLORS):
        if confidence > 0:
            pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def pose_keypoint_frame(people: Sequence[np.ndarray], width: int, height: int) -> dict:
    """One image in the POSE_KEYPOINT format of comfyui_controlnet_aux (OpenPose JSON, pixels)."""
    return {
        "people": [
            {
                "pose_keypoints_2d": [float(v) for v in np.asarray(person, dtype=np.float64).reshape(-1)],
                "face_keypoints_2d": None,
                "hand_left_keypoints_2d": None,
                "hand_right_keypoints_2d": None,
            }
            for person in people
        ],
        "canvas_width": int(width),
        "canvas_height": int(height),
    }


def keypoints_per_view(view_set, include_face: bool = True) -> List[np.ndarray]:
    """:func:`coco_keypoints` of every view of ``view_set``.

    The face keypoints need the views rendered first: rendering a view
    (:func:`.pipeline.render_view`) records which face features it shows.
    """
    face_ids = face_vertex_ids() if include_face else None
    return [coco_keypoints(view_set, i, include_face, face_ids) for i in range(len(view_set.specs))]


def pose_frames(view_set, per_view: Sequence[np.ndarray]) -> List[dict]:
    """POSE_KEYPOINT frames: one per view for a batch, one with a person per view for a sheet."""
    width, height = view_set.width, view_set.height
    if view_set.layout == "sheet":
        # Each view's keypoints move by its offset in the sheet; absent ones stay (0, 0, 0).
        shifted = [k + np.array([i * width, 0.0, 0.0]) * (k[:, 2:3] > 0) for i, k in enumerate(per_view)]
        return [pose_keypoint_frame(shifted, width * len(per_view), height)]
    return [pose_keypoint_frame([k], width, height) for k in per_view]


def joints_report(view_set, per_view: Sequence[np.ndarray]) -> dict:
    """The ``joints_json`` report: per view, the camera, the COCO keypoints and every body joint."""
    width, height = view_set.width, view_set.height
    return {
        "model_type": view_set.body.model_type,
        "gender": view_set.body.gender,
        "layout": view_set.layout,
        "views": [
            {
                "camera": views.describe(camera, spec, view_set.pivot, width, height),
                "offset_x": i * width if view_set.layout == "sheet" else 0,
                "coco_keypoints": [
                    {"name": name, "x": round(float(x), 3), "y": round(float(y), 3), "confidence": float(c)}
                    for name, (x, y, c) in zip(COCO_NAMES, per_view[i])
                ],
                "joints": body_joints_2d(view_set, i),
            }
            for i, (spec, camera) in enumerate(zip(view_set.specs, view_set.cameras))
        ],
    }
