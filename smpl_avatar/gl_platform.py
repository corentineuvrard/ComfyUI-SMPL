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
another package already imported ``OpenGL`` earlier, a warning is logged
because the choice can no longer take effect.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Optional, Tuple

LOGGER = logging.getLogger(__name__)

ENV_VAR = "PYOPENGL_PLATFORM"

_configured: Optional[Tuple[Optional[str], str]] = None


def decide(environ: Mapping[str, str], platform: str) -> Tuple[Optional[str], str]:
    """Return ``(value_to_set, reason)`` without side effects.

    ``value_to_set`` is ``None`` when the environment must be left as is.
    """
    current = environ.get(ENV_VAR)
    if current:
        return None, f"{ENV_VAR}={current!r} already set; respecting it"
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

    value, reason = decide(os.environ, sys.platform)
    if value is not None:
        if "OpenGL" in sys.modules:
            LOGGER.warning(
                "[SMPL Avatar] PyOpenGL was imported before this node pack, so "
                "%s=%s cannot take effect any more. Set %s=%s in the environment "
                "that launches ComfyUI if rendering fails.",
                ENV_VAR, value, ENV_VAR, value,
            )
        else:
            os.environ[ENV_VAR] = value
    LOGGER.info("[SMPL Avatar] OpenGL platform: %s", reason)
    _configured = (value, reason)
    return _configured
