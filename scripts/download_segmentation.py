"""Download the SMPL 24-part vertex segmentation ahead of time.

The node pack downloads it automatically the first time part masks are
requested; run this script instead if ComfyUI will run offline. The file is
published by Meshcapade GmbH (see THIRD_PARTY_NOTICES.md); it is fetched from
a pinned commit and checked against a SHA-256 digest.

Usage::

    python scripts/download_segmentation.py [--output PATH] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from smpl_avatar import paths, segmentation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=paths.SEGMENTATION_PATH)
    parser.add_argument("--force", action="store_true", help="download again even if a file is present")
    args = parser.parse_args(argv)
    if args.output.is_file() and not args.force:
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        if digest == segmentation.SEGMENTATION_SHA256:
            print(f"Already present: {args.output} (SHA-256 verified)")
            return 0
        print(f"error: {args.output} is present but is not the pinned file (SHA-256 {digest}, "
              f"expected {segmentation.SEGMENTATION_SHA256}); run again with --force to replace it")
        return 1
    print(f"Downloading {segmentation.SEGMENTATION_URL}")
    try:
        path = segmentation.download(args.output)
    except Exception as exc:  # network errors, a wrong checksum, permissions ...
        print(f"error: {exc}")
        return 1
    print(f"Saved to {path} (SHA-256 {segmentation.SEGMENTATION_SHA256} verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
