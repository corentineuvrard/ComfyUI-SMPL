"""SMPL 24-part vertex segmentation.

The segmentation file (``smpl_vert_segmentation.json``) is published by
Meshcapade GmbH in its body-modelling wiki repository. It is **not**
redistributed with this node pack because the repository carries no license
statement. It is downloaded on first use from the pinned commit below and
verified by SHA-256, or it can be placed manually at
``models/segmentation/smpl_vert_segmentation.json``.

The file maps each of the 24 part names to a list of SMPL vertex indices.
Vertices on part boundaries are listed in two or three parts. Faces are given a
single part by :func:`face_part_labels`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import paths

LOGGER = logging.getLogger(__name__)

SEGMENTATION_COMMIT = "be6a56e7a84f85339c0801da50b3ac028833c807"
SEGMENTATION_URL = (
    "https://raw.githubusercontent.com/Meshcapade/wiki/"
    f"{SEGMENTATION_COMMIT}/assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json"
)
SEGMENTATION_SHA256 = "0af78b5c581a225998b4804b4c9dd8351e155b34253947eb2541160852105ab8"
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024  # the real file is about 110 KB
ENV_NO_DOWNLOAD = "SMPL_AVATAR_NO_DOWNLOAD"

NUM_VERTICES = 6890

# Canonical part order. It follows the SMPL kinematic tree (joint 0..23), so
# index i is the part driven by SMPL joint i. The names are the keys used in
# the segmentation file.
PART_NAMES: Tuple[str, ...] = (
    "hips",            # 0  pelvis
    "leftUpLeg",       # 1  left thigh
    "rightUpLeg",      # 2  right thigh
    "spine",           # 3  lower back / belly
    "leftLeg",         # 4  left shin
    "rightLeg",        # 5  right shin
    "spine1",          # 6  mid torso
    "leftFoot",        # 7
    "rightFoot",       # 8
    "spine2",          # 9  chest / upper back
    "leftToeBase",     # 10 left toes
    "rightToeBase",    # 11 right toes
    "neck",            # 12
    "leftShoulder",    # 13 left collar / trapezius
    "rightShoulder",   # 14 right collar / trapezius
    "head",            # 15
    "leftArm",         # 16 left upper arm
    "rightArm",        # 17 right upper arm
    "leftForeArm",     # 18
    "rightForeArm",    # 19
    "leftHand",        # 20 left palm
    "rightHand",       # 21 right palm
    "leftHandIndex1",  # 22 left fingers
    "rightHandIndex1", # 23 right fingers
)
NUM_PARTS = len(PART_NAMES)

PART_DESCRIPTIONS: Dict[str, str] = {
    "hips": "pelvis",
    "leftUpLeg": "left thigh",
    "rightUpLeg": "right thigh",
    "spine": "lower torso (belly / lower back)",
    "leftLeg": "left shin and calf",
    "rightLeg": "right shin and calf",
    "spine1": "mid torso",
    "leftFoot": "left foot",
    "rightFoot": "right foot",
    "spine2": "chest and upper back",
    "leftToeBase": "left toes",
    "rightToeBase": "right toes",
    "neck": "neck",
    "leftShoulder": "left collarbone / trapezius area",
    "rightShoulder": "right collarbone / trapezius area",
    "head": "head",
    "leftArm": "left upper arm",
    "rightArm": "right upper arm",
    "leftForeArm": "left forearm",
    "rightForeArm": "right forearm",
    "leftHand": "left hand (palm)",
    "rightHand": "right hand (palm)",
    "leftHandIndex1": "left fingers",
    "rightHandIndex1": "right fingers",
}


class SegmentationUnavailableError(RuntimeError):
    """Raised when the segmentation file is missing and cannot be downloaded."""


def parse_segmentation(data: Mapping[str, Sequence[int]], num_vertices: int = NUM_VERTICES) -> np.ndarray:
    """Validate the JSON content and return a ``(num_vertices, 24)`` bool membership matrix."""
    keys = set(data)
    expected = set(PART_NAMES)
    if keys != expected:
        raise ValueError(
            f"segmentation parts mismatch: missing {sorted(expected - keys)}, "
            f"unexpected {sorted(keys - expected)}"
        )
    membership = np.zeros((num_vertices, NUM_PARTS), dtype=bool)
    for part_index, name in enumerate(PART_NAMES):
        indices = np.asarray(data[name])
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"part {name!r} must be a flat list of integers")
        if indices.size and (indices.min() < 0 or indices.max() >= num_vertices):
            raise ValueError(f"part {name!r} has vertex indices outside 0..{num_vertices - 1}")
        membership[indices, part_index] = True
    uncovered = np.flatnonzero(~membership.any(axis=1))
    if uncovered.size:
        raise ValueError(f"{uncovered.size} vertices belong to no part")
    return membership


def face_part_labels(membership: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Assign exactly one part to each face.

    Rule: the part containing the most of the face's three vertices wins. Ties
    (frequent on boundaries, where vertices belong to several parts) are broken
    by the number of the face's vertices that belong *only* to that part, then
    by the lowest part index.
    """
    membership = np.asarray(membership, dtype=bool)
    faces = np.asarray(faces, dtype=np.int64)
    exclusive = membership & (membership.sum(axis=1) == 1)[:, None]
    votes = membership[faces].sum(axis=1).astype(np.int64)  # (F, 24) in 0..3
    exclusive_votes = exclusive[faces].sum(axis=1).astype(np.int64)
    return np.argmax(votes * 4 + exclusive_votes, axis=1)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(destination: Path = paths.SEGMENTATION_PATH, url: str = SEGMENTATION_URL, timeout: float = 60.0) -> Path:
    """Download the pinned segmentation file, verify it and write it atomically."""
    LOGGER.info(
        "[SMPL Avatar] Downloading the SMPL part segmentation published by Meshcapade GmbH "
        "(%s). See THIRD_PARTY_NOTICES.md.", url,
    )
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise SegmentationUnavailableError(
            f"the response from {url} is larger than {MAX_DOWNLOAD_BYTES} bytes; this is not "
            "the expected 110 KB segmentation file (a captive portal or proxy page?)"
        )
    digest = _sha256(payload)
    if digest != SEGMENTATION_SHA256:
        raise SegmentationUnavailableError(
            f"the downloaded segmentation has SHA-256 {digest}, expected {SEGMENTATION_SHA256}: "
            "the file served by the pinned URL is not the expected one (a proxy or captive "
            "portal may have replaced it). Check your network, or download it manually from "
            f"{url}, verify the hash, and save it as {destination}."
        )
    parse_segmentation(json.loads(payload.decode("utf-8")))  # verified bytes: safe to parse
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        tmp.write_bytes(payload)
        tmp.replace(destination)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return destination


def ensure_file(path: Path = paths.SEGMENTATION_PATH, allow_download: Optional[bool] = None) -> Path:
    """Return the segmentation path, downloading it first if needed and allowed."""
    path = Path(path)
    if path.is_file():
        return path
    if allow_download is None:
        allow_download = os.environ.get(ENV_NO_DOWNLOAD, "").strip().lower() not in ("1", "true", "yes")
    if not allow_download:
        raise SegmentationUnavailableError(
            f"{path} is missing and downloads are disabled ({ENV_NO_DOWNLOAD}). "
            f"Download it manually from {SEGMENTATION_URL}"
        )
    try:
        return download(path)
    except SegmentationUnavailableError:
        raise
    except Exception as exc:  # network errors, permissions, ...
        raise SegmentationUnavailableError(
            f"Could not download the SMPL part segmentation ({exc}). Download it manually "
            f"from {SEGMENTATION_URL} and save it as {path}, or disable part masks."
        ) from exc


_cache_lock = threading.Lock()
_membership_cache: Dict[Path, np.ndarray] = {}


def load_membership(path: Optional[Path] = None) -> np.ndarray:
    """Load (and cache) the vertex membership matrix, downloading the file if needed."""
    path = ensure_file(Path(path) if path else paths.SEGMENTATION_PATH)
    with _cache_lock:
        if path not in _membership_cache:
            try:
                payload = path.read_bytes()
                membership = parse_segmentation(json.loads(payload.decode("utf-8")))
            except (ValueError, OSError) as exc:  # JSON / encoding / content errors
                raise SegmentationUnavailableError(
                    f"the SMPL part segmentation file {path} is unreadable ({type(exc).__name__}: "
                    f"{exc}). Delete it so that it is downloaded again, or replace it with the "
                    f"file from {SEGMENTATION_URL} (SHA-256 {SEGMENTATION_SHA256})."
                ) from exc
            digest = _sha256(payload)
            if digest != SEGMENTATION_SHA256:
                # Well formed but not the pinned file (edited, or another revision
                # of the upstream file): the masks follow whatever it contains.
                LOGGER.warning(
                    "[SMPL Avatar] %s is not the pinned SMPL part segmentation (SHA-256 %s, "
                    "expected %s): the part masks follow the segmentation it contains. Delete "
                    "the file to download the pinned one again.", path, digest, SEGMENTATION_SHA256,
                )
            _membership_cache[path] = membership
        return _membership_cache[path]
