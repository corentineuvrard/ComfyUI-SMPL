"""OpenGL platform selection for pyrender.

PyOpenGL reads the ``PYOPENGL_PLATFORM`` environment variable once, when the
``OpenGL`` package is first imported, and picks its windowing backend from it
(``egl``, ``osmesa``, or the platform default used through pyglet).

Policy implemented here:

1. If ``PYOPENGL_PLATFORM`` is already set (by the user, a launcher script or a
   container image), it is left untouched. This module never overwrites or
   removes it.
2. If it is not set and the process runs on Linux without a display server
   (neither ``DISPLAY`` nor ``WAYLAND_DISPLAY`` is set), it is set to ``egl`` so
   that pyrender can render headless on a GPU.
3. In every other case (Windows, macOS, Linux desktop session) nothing is set
   and pyrender uses its default pyglet backend.

The variable only has an effect if it is set before ``OpenGL`` is imported, so
:func:`configure` is called when the node pack is imported by ComfyUI. If
another package already imported ``OpenGL`` earlier, the choice can no longer
take effect in this process and rendering uses the worker described below.

Render worker
-------------
When OpenGL was already imported by someone else (recent ComfyUI versions do
so at startup, see below), rendering runs in a separate worker process whose
environment is computed by :func:`worker_environment`:

* ``SMPL_AVATAR_GL_PLATFORM``, if set, is used for the worker's
  ``PYOPENGL_PLATFORM`` (``default`` means "leave it unset").
* Otherwise the value of ``PYOPENGL_PLATFORM`` is passed on unchanged, with
  one exception: ComfyUI's built-in GLSL node (``comfy_extras/nodes_glsl.py``)
  loads ANGLE and calls ``os.environ.setdefault("PYOPENGL_PLATFORM", "egl")``
  at startup. On Windows and macOS that ``egl`` targets ANGLE, which only
  provides OpenGL ES and cannot run pyrender, so when ANGLE is loaded
  (``comfy_angle`` imported) on those platforms the worker does not inherit it.
* Then the same headless-Linux rule as above applies.

The ComfyUI process's own environment is never modified by this.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Collection, Dict, Mapping, Optional, Tuple

LOGGER = logging.getLogger(__name__)

ENV_VAR = "PYOPENGL_PLATFORM"
ENV_WORKER_OVERRIDE = "SMPL_AVATAR_GL_PLATFORM"

_configured: Optional[Tuple[Optional[str], str]] = None


def decide(environ: Mapping[str, str], platform: str) -> Tuple[Optional[str], str]:
    """Return ``(value_to_set, reason)`` without side effects.

    ``value_to_set`` is ``None`` when the environment must be left as is.
    """
    current = environ.get(ENV_VAR)
    if current:
        return None, f"{ENV_VAR}={current!r} already set; kept in this process (the render worker picks its own backend)"
    if platform.startswith("linux"):
        if environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"):
            return None, "Linux with a display server; using the default pyglet backend"
        return "egl", "headless Linux (no DISPLAY/WAYLAND_DISPLAY); selecting EGL"
    return None, f"platform {platform!r}; using the default pyglet backend"


def configure() -> Tuple[Optional[str], str]:
    """Apply :func:`decide` to ``os.environ`` once per process."""
    global _configured
    if _configured is not None:
        return _configured

    if os.environ.get(ENV_VAR) == "":
        # An empty value means "unset" to us but PyOpenGL would reject it.
        del os.environ[ENV_VAR]
    value, reason = decide(os.environ, sys.platform)
    if value is not None:
        if "OpenGL" in sys.modules:
            LOGGER.info(
                "[SMPL Avatar] PyOpenGL was already imported by another package; rendering "
                "will run in a separate worker process with %s=%s.", ENV_VAR, value,
            )
        else:
            os.environ[ENV_VAR] = value
    LOGGER.info("[SMPL Avatar] OpenGL platform: %s", reason)
    _configured = (value, reason)
    return _configured


def set_by_comfyui_angle(environ: Mapping[str, str], platform: str, modules: Collection[str]) -> bool:
    """True when ``PYOPENGL_PLATFORM=egl`` comes from ComfyUI's ANGLE setup (not usable by pyrender)."""
    return (
        environ.get(ENV_VAR) == "egl"
        and "comfy_angle" in modules
        and not platform.startswith("linux")
    )


def worker_environment(
    environ: Mapping[str, str], platform: str, modules: Collection[str]
) -> Tuple[Dict[str, str], str]:
    """Environment for the render worker process and a human-readable reason."""
    env = dict(environ)
    if env.get(ENV_VAR) == "":
        env.pop(ENV_VAR)  # empty means unset (PyOpenGL would reject "")
    override = environ.get(ENV_WORKER_OVERRIDE, "").strip()
    if override:
        if override.lower() == "default":
            env.pop(ENV_VAR, None)
            return env, f"{ENV_WORKER_OVERRIDE}=default: platform default backend"
        env[ENV_VAR] = override
        return env, f"{ENV_WORKER_OVERRIDE}={override!r}"
    notes = []
    if set_by_comfyui_angle(environ, platform, modules):
        env.pop(ENV_VAR, None)
        notes.append(
            f"ignoring {ENV_VAR}=egl set by ComfyUI's GLSL node for ANGLE (OpenGL ES only)"
        )
    value, reason = decide(env, platform)
    if value is not None:
        env[ENV_VAR] = value
    notes.append(reason)
    return env, "; ".join(notes)
