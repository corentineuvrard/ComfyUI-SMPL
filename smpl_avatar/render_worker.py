"""Render worker process: runs :func:`smpl_avatar.render.rasterize` on request.

Started by :class:`smpl_avatar.render.RenderWorker` with the OpenGL platform
chosen for it (see ``gl_platform.worker_environment``). Protocol on
stdin/stdout: each message is an 8-byte little-endian length followed by an
``.npz`` payload (NumPy arrays only, loaded without pickle). The worker exits
when its stdin is closed.
"""

import logging
import os
import sys
from pathlib import Path


def main() -> int:
    # Keep stdout for the protocol only: route anything else printed by this
    # process (Python or native libraries) to stderr.
    channel_out = os.fdopen(os.dup(1), "wb", buffering=0)
    try:
        os.dup2(2, 1)
    except OSError:  # no usable stderr (e.g. parent started without a console)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
    sys.stdout = sys.stderr
    channel_in = sys.stdin.buffer

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    from smpl_avatar import render

    while True:
        try:
            payload = render._read_message(channel_in)
        except EOFError:
            break
        render._write_message(channel_out, render.handle_request(payload))
    render.RENDERER_CACHE.release()
    channel_out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
