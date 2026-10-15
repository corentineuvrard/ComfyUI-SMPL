"""Offscreen orthographic rendering with pyrender.

One colour pass produces the shaded image, the exact coverage (alpha resolved
from 4x MSAA) and the depth buffer. An optional flat segmentation pass
produces a per-pixel body-part label map.

The OpenGL work (:func:`rasterize`) runs either in this process or in a
persistent worker process (``render_worker.py``). The worker is used when
OpenGL was already imported by another package before this pack needed it:
PyOpenGL binds to one platform per process, and recent ComfyUI versions import
it at startup for their own GLSL node, bound to ANGLE (OpenGL ES only), which
pyrender cannot use. Set ``SMPL_AVATAR_RENDER_PROCESS`` to ``worker`` or
``inprocess`` to force a mode. All post-processing (masks, depth, part labels)
runs in the calling process with NumPy.

pyrender is imported lazily, after :mod:`.gl_platform` had a chance to choose
the OpenGL platform.
"""

from __future__ import annotations

import atexit
import io
import logging
import math
import os
import struct
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

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


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[SMPL Avatar]"

ENV_RENDER_PROCESS = "SMPL_AVATAR_RENDER_PROCESS"
ENV_RENDER_TIMEOUT = "SMPL_AVATAR_RENDER_TIMEOUT"
RENDER_MODES = ("auto", "worker", "inprocess")
DEFAULT_RENDER_TIMEOUT = 300.0  # seconds per worker reply
WORKER_SCRIPT = Path(__file__).resolve().parent / "render_worker.py"
WORKER_MAGIC = b"SMPLAVW1"  # first bytes a healthy worker writes on its stdout

# Whether OpenGL had been imported by another package before this module first
# imported pyrender (None until then).
_opengl_preloaded: Optional[bool] = None


def import_pyrender():
    """Import the vendored pyrender after configuring the OpenGL platform.

    pyrender is vendored (``smpl_avatar/_vendor/pyrender``, MIT) because the
    last PyPI release pins ``PyOpenGL==3.1.0``, which would downgrade the
    ``PyOpenGL>=3.1.8`` that ComfyUI itself requires.
    """
    global _opengl_preloaded
    if _opengl_preloaded is None:
        _opengl_preloaded = "OpenGL" in sys.modules
    gl_platform.configure()
    try:
        from ._vendor import pyrender
    except ImportError as exc:
        raise ImportError(
            f"The renderer could not be loaded ({exc}). Install the node pack requirements "
            "(pip install -r requirements.txt); on headless Linux see the README section "
            "'OpenGL platform and headless Linux'."
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
    """Keeps one ``pyrender.OffscreenRenderer``; it is recreated only when the size changes.

    An OpenGL context belongs to the thread that created it, so the renderer
    may only be used from that thread (ComfyUI runs nodes on a single
    executor thread; other callers can use the worker process).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._renderer = None
        self._size: Optional[Tuple[int, int]] = None
        self._thread: Optional[int] = None
        self.creations = 0

    @contextmanager
    def renderer(self, width: int, height: int) -> Iterator[object]:
        with self._lock:
            self._check_thread()
            if self._renderer is None or self._size != (width, height):
                self._release_locked()
                pyrender = import_pyrender()
                self._renderer = pyrender.OffscreenRenderer(width, height)
                self._size = (width, height)
                self._thread = threading.get_ident()
                self.creations += 1
                _log_gl_renderer()
            yield self._renderer

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        return self._size

    def release(self) -> None:
        with self._lock:
            if self._renderer is not None and self._thread != threading.get_ident():
                # Deleting from another thread fails inside OpenGL, and so would
                # dropping the last reference (pyrender's __del__ calls delete()):
                # keep the renderer alive; the OS reclaims the context at exit.
                _ORPHANED.append(self._renderer)
                self._renderer = None
                self._size = None
                return
            self._release_locked()

    def _check_thread(self) -> None:
        if self._renderer is not None and self._thread != threading.get_ident():
            raise RuntimeError(
                "in-process rendering must stay on the thread that created the OpenGL "
                f"context; render from that thread or set {ENV_RENDER_PROCESS}=worker"
            )

    def _release_locked(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.delete()
            finally:
                self._renderer = None
                self._size = None


_gl_renderer_logged = False


def _log_gl_renderer() -> None:
    """Log once per process which GPU / driver renders (useful in bug reports)."""
    global _gl_renderer_logged
    if _gl_renderer_logged:
        return
    try:
        from OpenGL.GL import GL_RENDERER, GL_VERSION, glGetString

        LOGGER.info(
            "[SMPL Avatar] OpenGL renderer: %s (%s)",
            glGetString(GL_RENDERER).decode(errors="replace"),
            glGetString(GL_VERSION).decode(errors="replace"),
        )
        _gl_renderer_logged = True
    except Exception:  # pragma: no cover - purely informational
        pass


# Renderers released from a thread other than their own are kept here, never deleted.
_ORPHANED: list = []
RENDERER_CACHE = RendererCache()
# Releasing at exit avoids PyOpenGL / pyglet teardown noise during interpreter shutdown.
atexit.register(RENDERER_CACHE.release)


def rasterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    view: OrthoView,
    face_labels: Optional[np.ndarray] = None,
    cache: RendererCache = RENDERER_CACHE,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run the OpenGL passes in this process.

    Returns the RGBA colour pass ``(H, W, 4) uint8``, pyrender's depth output
    ``(H, W) float32`` and, if ``face_labels`` is given, the flat label pass
    ``(H, W, 3) uint8``.
    """
    pyrender = import_pyrender()
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
    return (
        np.ascontiguousarray(color, dtype=np.uint8),
        np.ascontiguousarray(raw_depth, dtype=np.float32),
        None if seg_rgb is None else np.ascontiguousarray(seg_rgb[..., :3], dtype=np.uint8),
    )


# ------------------------------------------------------------ render worker


def _pack(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _unpack(payload: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _write_message(stream, payload: bytes) -> None:
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("render worker channel closed")
        chunks += chunk
    return bytes(chunks)


def _read_message(stream) -> bytes:
    (size,) = struct.unpack("<Q", _read_exact(stream, 8))
    return _read_exact(stream, size)


def encode_request(vertices, faces, width, height, view: OrthoView, face_labels=None) -> bytes:
    arrays = {
        "vertices": np.asarray(vertices, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.uint32),
        "size": np.array([width, height], dtype=np.int64),
        "view": np.array(
            [view.center_x, view.center_y, view.xmag, view.ymag, view.camera_z, view.znear, view.zfar],
            dtype=np.float64,
        ),
    }
    if face_labels is not None:
        arrays["face_labels"] = np.asarray(face_labels, dtype=np.int64)
    return _pack(**arrays)


def handle_request(payload: bytes, cache: RendererCache = RENDERER_CACHE) -> bytes:
    """Worker side: decode a request, rasterize, encode the reply (errors included)."""
    try:
        request = _unpack(payload)
        width, height = (int(x) for x in request["size"])
        view = OrthoView(*(float(x) for x in request["view"]))
        color, raw_depth, seg = rasterize(
            request["vertices"], request["faces"], width, height, view,
            face_labels=request.get("face_labels"), cache=cache,
        )
        reply = {"color": color, "raw_depth": raw_depth}
        if seg is not None:
            reply["seg"] = seg
        return _pack(**reply)
    except Exception as exc:  # reported to the client; the worker keeps running
        return _pack(error=np.array(f"{type(exc).__name__}: {exc}"))


def render_timeout() -> Optional[float]:
    """Seconds to wait for one worker reply ($SMPL_AVATAR_RENDER_TIMEOUT; 0 = no limit)."""
    raw = os.environ.get(ENV_RENDER_TIMEOUT, "").strip()
    if not raw:
        return DEFAULT_RENDER_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{ENV_RENDER_TIMEOUT} must be a number of seconds, got {raw!r}") from None
    return value if value > 0 else None


def _forward_stderr(stream, logger=LOGGER) -> None:
    """Relay the worker's stderr (its logs, driver messages, tracebacks) to our logger."""
    with stream:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith(LOG_PREFIX + " "):
                line = line[len(LOG_PREFIX) + 1:]
            logger.info("%s render worker: %s", LOG_PREFIX, line)


class RenderWorker:
    """Persistent child process running :func:`rasterize`, started on first use.

    Protocol safety: a healthy worker writes :data:`WORKER_MAGIC` as the very
    first bytes of its stdout, so anything printed earlier by the interpreter
    start-up (a ``.pth`` file, ``sitecustomize``) is detected instead of being
    parsed as a reply. A reply that does not arrive within
    :func:`render_timeout` seconds gets the worker killed and restarted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._handshaken = False
        self.starts = 0

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start(self) -> None:
        env, reason = gl_platform.worker_environment(os.environ, sys.platform, set(sys.modules))
        LOGGER.info("%s Starting the render worker process (OpenGL: %s)", LOG_PREFIX, reason)
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, **kwargs,
        )
        self._handshaken = False
        threading.Thread(
            target=_forward_stderr, args=(self._process.stderr,),
            name="smpl-avatar-render-worker-stderr", daemon=True,
        ).start()
        self.starts += 1

    def rasterize(self, vertices, faces, width, height, view, face_labels=None):
        request = encode_request(vertices, faces, width, height, view, face_labels)
        timeout = render_timeout()
        with self._lock:
            if not self.running:
                self._start()
            process = self._process
            timed_out = threading.Event()

            def _kill_on_timeout() -> None:
                timed_out.set()
                try:
                    process.kill()
                except OSError:
                    pass

            timer = threading.Timer(timeout, _kill_on_timeout) if timeout else None
            if timer is not None:
                timer.daemon = True
                timer.start()
            try:
                _write_message(process.stdin, request)
                if not self._handshaken:
                    head = _read_exact(process.stdout, len(WORKER_MAGIC))
                    if head != WORKER_MAGIC:
                        raise OSError(
                            f"unexpected output on the render worker's stdout: {head!r} "
                            "(something printed during the interpreter start-up?)"
                        )
                    self._handshaken = True
                reply = _unpack(_read_message(process.stdout))
            except (OSError, EOFError, ValueError) as exc:
                self._process = None
                _shutdown(process, timeout=1.0)
                if timed_out.is_set():
                    raise RuntimeError(
                        f"the SMPL Avatar render worker did not answer within {timeout:g} s and "
                        f"was killed (set {ENV_RENDER_TIMEOUT} to raise the limit for very large "
                        "renders). It is restarted on the next render."
                    ) from exc
                raise RuntimeError(
                    f"the SMPL Avatar render worker stopped (exit code {process.returncode}): {exc}. "
                    "Its output was logged above (ComfyUI log). It is restarted on the next render."
                ) from exc
            finally:
                if timer is not None:
                    timer.cancel()
        if "error" in reply:
            raise RuntimeError(f"render worker: {reply['error']}")
        return reply["color"], reply["raw_depth"], reply.get("seg")

    def close(self, timeout: float = 5.0) -> None:
        # A render blocked on a hung worker holds the lock: never wait for it
        # forever (this also runs at exit). Killing the worker unblocks it.
        acquired = self._lock.acquire(timeout=timeout)
        try:
            process, self._process = self._process, None
        finally:
            if acquired:
                self._lock.release()
        if process is not None:
            _shutdown(process, timeout if acquired else 0.0)


def _shutdown(process: subprocess.Popen, timeout: float) -> None:
    """Ask the worker to exit (EOF on stdin), kill it if needed, close both pipes."""
    try:
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()


RENDER_WORKER = RenderWorker()
atexit.register(RENDER_WORKER.close)


def choose_mode(requested: Optional[str], opengl_preloaded: bool) -> str:
    """``auto`` picks the worker when OpenGL was already imported by another package."""
    mode = (requested or "auto").strip().lower()
    if mode not in RENDER_MODES:
        raise ValueError(f"{ENV_RENDER_PROCESS} must be one of {RENDER_MODES}, got {requested!r}")
    if mode == "auto":
        return "worker" if opengl_preloaded else "inprocess"
    return mode


def current_mode(requested: Optional[str] = None) -> str:
    """Rendering mode (``worker`` or ``inprocess``) for ``requested`` or $SMPL_AVATAR_RENDER_PROCESS."""
    preloaded = _opengl_preloaded if _opengl_preloaded is not None else "OpenGL" in sys.modules
    return choose_mode(requested or os.environ.get(ENV_RENDER_PROCESS), preloaded)


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
    mode: Optional[str] = None,
) -> RenderResult:
    """Render the mesh and derive image, masks, depth and part masks.

    ``mode`` forces ``worker`` or ``inprocess``; by default :func:`current_mode`
    decides. ``cache`` is only used in-process.
    """
    if background not in BACKGROUNDS:
        raise ValueError(f"unknown background {background!r}; expected one of {list(BACKGROUNDS)}")
    mode = current_mode(mode)
    if mode == "worker":
        color, raw_depth, seg_rgb = RENDER_WORKER.rasterize(vertices, faces, width, height, view, face_labels)
    else:
        color, raw_depth, seg_rgb = rasterize(vertices, faces, width, height, view, face_labels, cache)

    if color.shape[-1] == 3:
        # Platforms without framebuffer objects (pyrender's OSMesa backend) return
        # RGB only: fall back to a binary coverage taken from the depth buffer.
        alpha8 = np.where(np.asarray(raw_depth) > 0, 255, 0).astype(np.uint8)
        color = np.dstack([color, alpha8])
    alpha8 = color[..., 3]
    coverage = alpha8.astype(np.float32) / 255.0
    # Hard mask: at least half of the 4 MSAA samples on the body. A pixel with
    # exactly 2 samples resolves to alpha 127 or 128 depending on the driver,
    # so the threshold sits below both values (any value in 65..127 works).
    mask = coverage if soft_mask else (alpha8 >= 127).astype(np.float32)

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
