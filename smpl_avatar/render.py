"""Offscreen orthographic rendering with pyrender.

One colour pass produces the shaded image, the exact coverage (alpha resolved
from 4x MSAA) and the depth buffer. An optional flat segmentation pass
produces a per-pixel body-part label map.

pyrender is imported lazily, after :mod:`.gl_platform` had a chance to choose
the OpenGL platform.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from . import gl_platform
from .masks import fill_labels, fill_values, labels_to_part_masks
from .segmentation import NUM_PARTS

BACKGROUNDS = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0), "gray": (0.5, 0.5, 0.5)}
FRAMINGS = ("fit", "fixed")
BODY_COLOR = (0.72, 0.72, 0.74)
# Depth map encoding: nearest visible point = 1.0, farthest visible body
# point = DEPTH_FAR_VALUE, background = 0.0.
DEPTH_FAR_VALUE = 0.2


def import_pyrender():
    """Import pyrender after configuring the OpenGL platform."""
    gl_platform.configure()
    if not hasattr(np, "infty"):
        # pyrender 0.1.45 still references np.infty, which NumPy 2.0 removed.
        np.infty = np.inf
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError(
            f"pyrender could not be imported ({exc}). Install the node pack requirements "
            "(pip install -r requirements.txt); on headless Linux see the README section "
            "'Headless Linux (EGL)'."
        ) from exc
    return pyrender


# ------------------------------------------------------------------- camera


@dataclass(frozen=True)
class OrthoView:
    """Orthographic camera looking down -z, centred on ``(center_x, center_y)``.

    The view spans ``[center_x - xmag, center_x + xmag]`` horizontally and
    ``[center_y - ymag, center_y + ymag]`` vertically, in metres, so one pixel
    covers ``2 * ymag / height`` metres in both directions.
    """

    center_x: float
    center_y: float
    xmag: float
    ymag: float
    camera_z: float
    znear: float
    zfar: float

    def camera_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = (self.center_x, self.center_y, self.camera_z)
        return pose

    def project(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """World ``(N, 3)`` points -> pixel ``(N, 2)`` coordinates (x right, y down)."""
        points = np.asarray(points, dtype=np.float64)
        u = (points[:, 0] - (self.center_x - self.xmag)) / (2 * self.xmag) * width
        v = ((self.center_y + self.ymag) - points[:, 1]) / (2 * self.ymag) * height
        return np.stack([u, v], axis=1)

    def distance_from_raw_depth(self, raw: np.ndarray) -> np.ndarray:
        """Convert pyrender's depth output to linear camera distance (0 = background).

        pyrender linearises the depth buffer with the perspective formula for
        every camera type. For an orthographic projection the NDC depth is
        linear in distance, which gives ``d = n + f - n * f / raw``.
        """
        raw = np.asarray(raw, dtype=np.float64)
        distance = np.zeros_like(raw)
        covered = raw > 0
        distance[covered] = self.znear + self.zfar - self.znear * self.zfar / raw[covered]
        return distance


def compute_view(
    vertices: np.ndarray,
    width: int,
    height: int,
    framing: str = "fit",
    padding: float = 0.1,
    view_height_m: float = 2.2,
) -> OrthoView:
    """Place the orthographic camera.

    * ``fit``: the body bounding box is centred and scaled to fill the image,
      with ``padding`` extra margin (0.1 = 10 % of the half extent).
    * ``fixed``: the view is ``view_height_m`` metres tall whatever the body
      size, so avatars of different heights keep a common scale. The feet sit
      ``padding`` times the half view height above the bottom edge.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    v = np.asarray(vertices, dtype=np.float64)
    (xmin, ymin, zmin), (xmax, ymax, zmax) = v.min(axis=0), v.max(axis=0)
    aspect = width / height
    center_x = (xmin + xmax) / 2.0
    if framing == "fit":
        ymag = max((ymax - ymin) / 2.0, (xmax - xmin) / 2.0 / aspect) * (1.0 + padding)
        center_y = (ymin + ymax) / 2.0
    elif framing == "fixed":
        if view_height_m <= 0:
            raise ValueError("view_height_m must be positive")
        ymag = view_height_m / 2.0
        center_y = ymin - padding * ymag + ymag
    else:
        raise ValueError(f"unknown framing {framing!r}; expected one of {FRAMINGS}")
    camera_z = zmax + 1.0
    return OrthoView(
        center_x=float(center_x),
        center_y=float(center_y),
        xmag=float(ymag * aspect),
        ymag=float(ymag),
        camera_z=float(camera_z),
        znear=0.5,
        zfar=float(camera_z - zmin + 0.5),
    )


def rotate_y(vertices: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate points about the vertical axis (positive = counter-clockwise seen from above)."""
    if not degrees:
        return np.asarray(vertices)
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.asarray(vertices) @ rotation.T


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals = np.zeros_like(v)
    for k in range(3):
        np.add.at(normals, f[:, k], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-12)).astype(np.float32)


def _look_along(direction) -> np.ndarray:
    """4x4 pose whose -z axis points along ``direction`` (for directional lights)."""
    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    z = -forward
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    return pose


# ---------------------------------------------------------- segmentation code


def segmentation_palette() -> np.ndarray:
    """24 RGB colours used by the flat label pass.

    Red/green lie on a circle and blue is 255 for every part (the background is
    black). Colours in convex position guarantee that an MSAA blend of two or
    more parts never equals another part's colour, and any blend with the
    background has blue < 255, so an exact colour match identifies a pixel
    fully covered by a single part.
    """
    angles = 2.0 * np.pi * np.arange(NUM_PARTS) / NUM_PARTS
    red = np.round(128 + 120 * np.cos(angles))
    green = np.round(128 + 120 * np.sin(angles))
    blue = np.full(NUM_PARTS, 255.0)
    return np.stack([red, green, blue], axis=1).astype(np.uint8)


def decode_labels(seg_rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Label map from the flat segmentation pass; ``-1`` outside ``region``.

    Exact palette matches are trusted. Remaining pixels of ``region`` (anti-
    aliased boundaries) take the majority label of their neighbours; isolated
    leftovers use the nearest palette colour after removing the background
    blend.
    """
    palette = segmentation_palette().astype(np.int64)
    rgb = np.asarray(seg_rgb)[..., :3].astype(np.int64)
    codes = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    palette_codes = (palette[:, 0] << 16) | (palette[:, 1] << 8) | palette[:, 2]
    order = np.argsort(palette_codes)
    sorted_codes = palette_codes[order]
    position = np.clip(np.searchsorted(sorted_codes, codes), 0, NUM_PARTS - 1)
    exact = sorted_codes[position] == codes
    labels = np.where(exact, order[position], -1)
    region = np.asarray(region, dtype=bool)
    labels[~region] = -1
    labels = fill_labels(labels, region)

    leftover = region & (labels < 0)
    if leftover.any():
        pixels = rgb[leftover].astype(np.float64)
        scale = np.maximum(pixels[:, 2:3], 1.0) / 255.0
        unblended = pixels[:, :2] / scale
        distances = ((unblended[:, None, :] - palette[None, :, :2]) ** 2).sum(axis=2)
        labels[leftover] = distances.argmin(axis=1)
    return labels


# ------------------------------------------------------------------ renderer


class RendererCache:
    """Keeps one ``pyrender.OffscreenRenderer``; it is recreated only when the size changes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._renderer = None
        self._size: Optional[Tuple[int, int]] = None
        self.creations = 0

    @contextmanager
    def renderer(self, width: int, height: int) -> Iterator[object]:
        with self._lock:
            if self._renderer is None or self._size != (width, height):
                self._release_locked()
                pyrender = import_pyrender()
                self._renderer = pyrender.OffscreenRenderer(width, height)
                self._size = (width, height)
                self.creations += 1
            yield self._renderer

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        return self._size

    def release(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.delete()
            finally:
                self._renderer = None
                self._size = None


RENDERER_CACHE = RendererCache()


@dataclass
class RenderResult:
    image: np.ndarray  # (H, W, 3) float32, composited over the background
    mask: np.ndarray  # (H, W) float32, hard (0/1) or soft coverage
    depth: np.ndarray  # (H, W) float32, near = 1, far body = DEPTH_FAR_VALUE, background = 0
    distance: np.ndarray  # (H, W) float32, camera distance in metres (0 = background)
    coverage: np.ndarray  # (H, W) float32, MSAA coverage in [0, 1]
    labels: Optional[np.ndarray] = None  # (H, W) int64 part index, -1 = background
    part_masks: Optional[np.ndarray] = None  # (24, H, W) float32


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    view: OrthoView,
    face_labels: Optional[np.ndarray] = None,
    background: str = "black",
    soft_mask: bool = False,
    cache: RendererCache = RENDERER_CACHE,
) -> RenderResult:
    pyrender = import_pyrender()
    if background not in BACKGROUNDS:
        raise ValueError(f"unknown background {background!r}; expected one of {list(BACKGROUNDS)}")
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)

    camera = pyrender.OrthographicCamera(
        xmag=view.xmag, ymag=view.ymag, znear=view.znear, zfar=view.zfar
    )
    camera_pose = view.camera_pose()

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.22, 0.22, 0.22])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*BODY_COLOR, 1.0), metallicFactor=0.0, roughnessFactor=0.7, alphaMode="OPAQUE"
    )
    body = pyrender.Primitive(
        positions=vertices, normals=vertex_normals(vertices, faces), indices=faces, material=material
    )
    scene.add(pyrender.Mesh(primitives=[body]))
    scene.add(camera, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.9), pose=_look_along((0.35, -0.45, -1.0)))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=0.6), pose=_look_along((-0.5, -0.1, -1.0)))

    seg_scene = seg_map = None
    if face_labels is not None:
        face_labels = np.asarray(face_labels)
        if face_labels.shape != (faces.shape[0],):
            raise ValueError("face_labels must have one entry per face")
        seg_scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        seg_scene.add(camera, pose=camera_pose)
        seg_map = {}
        for part, color in enumerate(segmentation_palette()):
            part_faces = faces[face_labels == part]
            if len(part_faces) == 0:
                continue
            node = seg_scene.add(pyrender.Mesh(primitives=[pyrender.Primitive(positions=vertices, indices=part_faces)]))
            seg_map[node] = color.astype(np.float64)

    with cache.renderer(width, height) as renderer:
        color, raw_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        seg_rgb = None
        if seg_scene is not None:
            seg_rgb, _ = renderer.render(seg_scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_map)

    color = np.asarray(color)
    alpha8 = color[..., 3]
    coverage = alpha8.astype(np.float32) / 255.0
    mask = coverage if soft_mask else (coverage >= 0.5).astype(np.float32)

    # pyrender's RGBA output is the MSAA average of body and a transparent black
    # background, i.e. premultiplied by coverage: composite with "over".
    bg = np.asarray(BACKGROUNDS[background], dtype=np.float32)
    image = color[..., :3].astype(np.float32) / 255.0 + (1.0 - coverage[..., None]) * bg
    image = np.clip(image, 0.0, 1.0)

    distance = view.distance_from_raw_depth(raw_depth)
    depth, distance = _depth_map(distance, alpha8, mask)

    labels = part_masks = None
    if seg_rgb is not None:
        labels = decode_labels(seg_rgb, region=alpha8 > 0)
        part_masks = labels_to_part_masks(labels, mask)

    return RenderResult(
        image=image.astype(np.float32),
        mask=mask.astype(np.float32),
        depth=depth,
        distance=distance,
        coverage=coverage,
        labels=labels,
        part_masks=part_masks,
    )


def _depth_map(distance: np.ndarray, alpha8: np.ndarray, mask: np.ndarray):
    """Normalise camera distance to a ControlNet-style depth image.

    Only fully covered pixels (all MSAA samples on the body) have a reliable
    depth value, because resolving a multisampled depth buffer is
    implementation-defined on partially covered pixels. Edge pixels take the
    mean distance of their fully covered neighbours.
    """
    interior = (alpha8 == 255) & (distance > 0)
    region = alpha8 > 0
    depth = np.zeros(distance.shape, dtype=np.float32)
    if not interior.any():
        return depth, np.zeros(distance.shape, dtype=np.float32)
    near = distance[interior].min()
    far = distance[interior].max()
    # Edge pixels that no fully covered neighbour reaches keep their own value,
    # or the far value when the resolved depth sample missed the body.
    base = np.where(distance > 0, distance, far)
    filled = fill_values(base, interior, region)
    filled = np.where(region, np.clip(filled, near, far), 0.0)
    span = max(far - near, 1e-6)
    value = 1.0 - (1.0 - DEPTH_FAR_VALUE) * (filled - near) / span
    depth = np.where(region, value, 0.0) * mask
    return depth.astype(np.float32), filled.astype(np.float32)
