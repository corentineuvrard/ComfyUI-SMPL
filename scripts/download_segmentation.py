"""Download the SMPL 24-part vertex segmentation ahead of time.

The node pack downloads it automatically the first time part masks are
requested; run this script instead if ComfyUI will run offline. The file is
published by Meshcapade GmbH (see THIRD_PARTY_NOTICES.md); it is fetched from
a pinned commit and checked against a SHA-256 digest.

Usage::

    python scripts/download_segmentation.py [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from smpl_avatar import paths, segmentation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=paths.SEGMENTATION_PATH)
    args = parser.parse_args(argv)
    if args.output.is_file():
        print(f"Already present: {args.output}")
        return 0
    print(f"Downloading {segmentation.SEGMENTATION_URL}")
    try:
        path = segmentation.ensure_file(args.output, allow_download=True)
    except segmentation.SegmentationUnavailableError as exc:
        print(f"error: {exc}")
        return 1
    print(f"Saved to {path} (SHA-256 {segmentation.SEGMENTATION_SHA256} verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
