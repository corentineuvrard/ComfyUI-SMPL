"""ComfyUI entry point of the SMPL Avatar node pack."""

from .smpl_avatar import gl_platform, paths

# Must run before anything imports PyOpenGL; see smpl_avatar/gl_platform.py.
gl_platform.configure()
paths.create_placeholders()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
