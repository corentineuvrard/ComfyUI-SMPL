"""Camera views around a body: presets, free angles and a framing shared by all views.

The camera stays put and looks down -z; a view turns the body instead, about
its own centre: first by ``azimuth`` about the vertical axis, then by
``elevation`` about the horizontal axis. With the conventions of
:func:`render.rotate_y`, azimuth 0 is the front view, +90 shows the body's
right side (right profile), 180 the back and -90 the left profile; a positive
elevation tilts the top of the body towards the camera, i.e. the camera looks
down from above.

All views of a set share one scale, so that the body keeps the same size on a
character sheet: the views are framed together, not one by one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .render import FRAMINGS, OrthoView, PerspectiveView, View

PROJECTIONS = ("orthographic", "perspective")

# preset -> ((label, azimuth in degrees), ...), in the order the views are output.
VIEW_PRESETS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "front": (("front", 0.0),),
    "back": (("back", 180.0),),
    "left profile": (("left profile", -90.0),),
    "right profile": (("right profile", 90.0),),
    "front 3/4 left": (("front 3/4 left", -45.0),),
    "front 3/4 right": (("front 3/4 right", 45.0),),
    "front / side / back": (("front", 0.0), ("right profile", 90.0), ("back", 180.0)),
    "turnaround 4": (("front", 0.0), ("right profile", 90.0), ("back", 180.0), ("left profile", 270.0)),
    "turnaround 8": (
        ("front", 0.0), ("front 3/4 right", 45.0), ("right profile", 90.0), ("back 3/4 right", 135.0),
        ("back", 180.0), ("back 3/4 left", 225.0), ("left profile", 270.0), ("front 3/4 left", 315.0),
    ),
    "custom": (("custom", 0.0),),
}


@dataclass(frozen=True)
class ViewSpec:
    label: str
    azimuth: float  # degrees, in (-180, 180]
    elevation: float  # degrees


def _wrap(angle: float) -> float:
    """Angle in degrees mapped to (-180, 180]."""
    wrapped = math.fmod(angle, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped + 0.0  # no -0.0


def expand(preset: str, azimuth: float = 0.0, elevation: float = 0.0) -> List[ViewSpec]:
    """The views of ``preset``; ``azimuth`` turns every view, ``elevation`` applies to all."""
    if preset not in VIEW_PRESETS:
        raise ValueError(f"unknown views {preset!r}; expected one of {list(VIEW_PRESETS)}")
    for name, value in (("azimuth", azimuth), ("elevation", elevation)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not -90.0 < elevation < 90.0:
        raise ValueError(f"elevation must be between -90 and 90 degrees (exclusive), got {elevation!r}")

    def label_of(label: str) -> str:
        offset = _wrap(azimuth)
        return label if offset == 0 or preset == "custom" else f"{label} {offset:+g}"

    return [ViewSpec(label_of(label), _wrap(angle + azimuth), float(elevation)) for label, angle in VIEW_PRESETS[preset]]


def rotation(spec: ViewSpec) -> np.ndarray:
    """3x3 rotation applied to the body for ``spec`` (azimuth about y, then elevation about x)."""
    a, e = math.radians(spec.azimuth), math.radians(spec.elevation)
    ca, sa, ce, se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
    yaw = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return pitch @ yaw


def pivot_of(vertices: np.ndarray) -> np.ndarray:
    """Centre of the bounding box: the point the views turn the body about."""
    v = np.asarray(vertices, dtype=np.float64)
    return (v.min(axis=0) + v.max(axis=0)) / 2.0


def transform(points: np.ndarray, spec: ViewSpec, pivot: np.ndarray) -> np.ndarray:
    """``points`` of the body as seen in ``spec`` (world coordinates, camera looking down -z)."""
    p = np.asarray(points, dtype=np.float64)
    return (p - pivot) @ rotation(spec).T + pivot


def frame_views(
    posed: Sequence[np.ndarray],
    width: int,
    height: int,
    projection: str = "orthographic",
    fov_deg: float = 30.0,
    framing: str = "fit",
    padding: float = 0.1,
    view_height_m: float = 2.2,
) -> List[View]:
    """One camera per entry of ``posed`` (the body as seen in each view), all at the same scale.

    * ``fit``: the scale is the largest one that keeps every view inside its
      image with ``padding`` margin (0.1 = 10 % of the extent). Each view is
      centred horizontally on its own body; all share one vertical centre (the
      feet are on exactly one line in orthographic views at elevation 0;
      otherwise the lowest point can differ from view to view).
    * ``fixed``: the view is ``view_height_m`` metres tall (measured at the
      body's depth for ``perspective``) and the lowest projected point of all
      views sits ``padding`` times the half view height above the bottom edge.
      A perspective camera never comes closer than the front of the body.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if projection not in PROJECTIONS:
        raise ValueError(f"unknown projection {projection!r}; expected one of {PROJECTIONS}")
    if framing not in FRAMINGS:
        raise ValueError(f"unknown framing {framing!r}; expected one of {FRAMINGS}")
    if not posed:
        raise ValueError("no view to frame")
    if not (math.isfinite(padding) and padding >= 0):
        raise ValueError(f"padding must be a finite number >= 0, got {padding!r}")
    if not (math.isfinite(view_height_m) and view_height_m > 0):
        raise ValueError(f"view_height_m must be a positive number, got {view_height_m!r}")
    if projection == "perspective" and not (math.isfinite(fov_deg) and 0.0 < fov_deg < 180.0):
        raise ValueError(f"fov must be between 0 and 180 degrees (exclusive), got {fov_deg!r}")

    points = [np.asarray(p, dtype=np.float64) for p in posed]
    lows = np.array([p.min(axis=0) for p in points])
    highs = np.array([p.max(axis=0) for p in points])
    aspect = width / height
    center_xs = (lows[:, 0] + highs[:, 0]) / 2.0
    y_low, y_high = lows[:, 1].min(), highs[:, 1].max()
    z_low, z_high = lows[:, 2].min(), highs[:, 2].max()

    if projection == "orthographic":
        if framing == "fit":
            half_x = ((highs[:, 0] - lows[:, 0]) / 2.0).max() / aspect
            ymag = max((y_high - y_low) / 2.0, half_x) * (1.0 + padding)
            center_y = (y_low + y_high) / 2.0
        else:
            ymag = view_height_m / 2.0
            center_y = y_low - padding * ymag + ymag
        camera_z = z_high + 1.0
        return [
            OrthoView(center_x=float(cx), center_y=float(center_y), xmag=float(ymag * aspect), ymag=float(ymag),
                      camera_z=float(camera_z), znear=0.5, zfar=float(camera_z - z_low + 0.5))
            for cx in center_xs
        ]

    yfov = math.radians(fov_deg)
    tan_y = math.tan(yfov / 2.0)
    tan_x = tan_y * aspect
    if framing == "fit":
        center_y = (y_low + y_high) / 2.0
        # The camera must be far enough for every point of every view to fall
        # inside the shrunken frustum: |x - cx| <= (D - z) tan_x / (1 + padding).
        need = max(
            float(np.max(p[:, 2] + np.maximum(np.abs(p[:, 0] - cx) / tan_x, np.abs(p[:, 1] - center_y) / tan_y)
                         * (1.0 + padding)))
            for p, cx in zip(points, center_xs)
        )
        camera_z = max(need, z_high + 0.05)
    else:
        half = view_height_m / 2.0
        body_z = (z_low + z_high) / 2.0
        camera_z = max(body_z + half / tan_y, z_high + 0.05)
        # The nearer a point, the larger it appears: put the lowest *projected*
        # point on the padding line, |y - cy| <= (D - z) tan_y (1 - padding).
        center_y = min(float(np.min(p[:, 1] + (camera_z - p[:, 2]) * tan_y * (1.0 - padding))) for p in points)
    znear = max(0.01, 0.5 * (camera_z - z_high))
    zfar = camera_z - z_low + 0.5
    return [
        PerspectiveView(center_x=float(cx), center_y=float(center_y), camera_z=float(camera_z), yfov=float(yfov),
                        aspect=float(aspect), znear=float(znear), zfar=float(zfar))
        for cx in center_xs
    ]


# Pixels. ``fit`` framing with padding = 0 puts the outermost vertices exactly
# on the border, where rounding can leave them a hair outside the image.
EDGE_TOLERANCE_PX = 1e-6


def outside_fraction(points: np.ndarray, view: View, width: int, height: int) -> float:
    """Fraction of ``points`` that project outside the image (0 = fully visible).

    A point within :data:`EDGE_TOLERANCE_PX` of the border counts as inside.
    """
    uv = view.project(points, width, height)
    finite = np.isfinite(uv).all(axis=1)
    tol = EDGE_TOLERANCE_PX
    outside = (~finite | (uv[:, 0] < -tol) | (uv[:, 0] > width + tol)
               | (uv[:, 1] < -tol) | (uv[:, 1] > height + tol))
    return float(outside.mean()) if len(uv) else 0.0


def describe(view: View, spec: ViewSpec, pivot: np.ndarray, width: int, height: int) -> dict:
    """Camera of one view as plain numbers (for the JSON output), in the OpenCV convention.

    World frame: the body's own (metres, y up, facing +z). ``world_to_camera``
    maps a body point to camera coordinates (x right, y down, z forward).
    Pixels (u right, v down) follow from ``intrinsics`` for a perspective
    view (u = fx x / z + cx), and from ``pixels_per_metre`` for an
    orthographic one (u = cx + s x, v = cy + s y). Pixel coordinates are
    measured from the top-left corner of the image, pixel i covering [i, i+1)
    (subtract 0.5 for OpenCV's pixel-centre convention).
    """
    r = rotation(spec)
    gl = np.eye(4)  # body point -> OpenGL camera coordinates (y up, looking down -z)
    gl[:3, :3] = r
    gl[:3, 3] = pivot - r @ pivot - np.array([view.center_x, view.center_y, view.camera_z])
    world_to_camera = np.diag([1.0, -1.0, -1.0, 1.0]) @ gl
    camera = {
        "label": spec.label,
        "azimuth_deg": spec.azimuth,
        "elevation_deg": spec.elevation,
        "projection": view.projection,
        "image_size": [int(width), int(height)],
        "pixel_origin": "top-left corner of the image",
        "world_to_camera": world_to_camera.round(6).tolist(),
    }
    if isinstance(view, OrthoView):
        camera["pixels_per_metre"] = round(height / (2.0 * view.ymag), 6)
        camera["principal_point"] = [width / 2.0, height / 2.0]
    else:
        focal = round(view.focal_px(height), 6)
        camera["fov_y_deg"] = round(math.degrees(view.yfov), 6)
        camera["intrinsics"] = [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    return camera
