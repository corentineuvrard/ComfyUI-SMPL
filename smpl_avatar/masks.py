"""Pure NumPy helpers for part label maps and part masks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .segmentation import NUM_PARTS, PART_NAMES

_TORSO = ("hips", "spine", "spine1", "spine2", "leftShoulder", "rightShoulder")
_ARMS = ("leftArm", "rightArm", "leftForeArm", "rightForeArm")
_HANDS = ("leftHand", "rightHand", "leftHandIndex1", "rightHandIndex1")
_LEGS = ("leftUpLeg", "rightUpLeg", "leftLeg", "rightLeg")
_FEET = ("leftFoot", "rightFoot", "leftToeBase", "rightToeBase")

PRESETS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "head": ("head",),
    "head and neck": ("head", "neck"),
    "torso": _TORSO,
    "arms": _ARMS,
    "hands": _HANDS,
    "arms and hands": _ARMS + _HANDS,
    "legs": _LEGS,
    "feet": _FEET,
    "legs and feet": _LEGS + _FEET,
    "upper body": ("head", "neck", "spine", "spine1", "spine2", "leftShoulder", "rightShoulder")
    + _ARMS + _HANDS,
    "lower body": ("hips",) + _LEGS + _FEET,
    "full body": PART_NAMES,
}

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def part_indices(names: Iterable[str]) -> List[int]:
    """Map part names to canonical indices (sorted, unique)."""
    lookup = {name: i for i, name in enumerate(PART_NAMES)}
    result = set()
    for name in names:
        if name not in lookup:
            raise KeyError(f"unknown SMPL part {name!r}")
        result.add(lookup[name])
    return sorted(result)


def union_part_masks(part_masks: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Union of selected parts.

    ``part_masks`` is ``(24 * N, H, W)``: N avatars, each contributing 24 masks in
    canonical :data:`PART_NAMES` order. Returns ``(N, H, W)``.
    """
    part_masks = np.asarray(part_masks)
    if part_masks.ndim != 3 or part_masks.shape[0] == 0 or part_masks.shape[0] % NUM_PARTS:
        raise ValueError(
            f"expected part masks of shape (24*N, H, W), got {part_masks.shape}; connect the "
            "'part_masks' output of 'Generate SMPL Avatar'"
        )
    count = part_masks.shape[0] // NUM_PARTS
    grouped = part_masks.reshape(count, NUM_PARTS, *part_masks.shape[1:])
    selected = sorted(set(int(i) for i in indices))
    if any(i < 0 or i >= NUM_PARTS for i in selected):
        raise ValueError(f"part indices must be in 0..{NUM_PARTS - 1}")
    if not selected:
        return np.zeros((count,) + part_masks.shape[1:], dtype=part_masks.dtype)
    return grouped[:, selected].max(axis=1)


def _shifted(array: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    out = np.full_like(array, fill)
    h, w = array.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[yd, xd] = array[ys, xs]
    return out


def fill_labels(labels: np.ndarray, region: np.ndarray, max_iterations: int = 64) -> np.ndarray:
    """Give unknown pixels (``-1``) inside ``region`` the most frequent label of their 8 neighbours.

    Repeated until every pixel of ``region`` connected to a known label is filled.
    Ties go to the lowest label.
    """
    labels = np.array(labels, dtype=np.int64, copy=True)
    region = np.asarray(region, dtype=bool)
    for _ in range(max_iterations):
        todo = region & (labels < 0)
        if not todo.any():
            break
        ys, xs = np.nonzero(todo)
        counts = np.zeros((ys.size, NUM_PARTS + 1), dtype=np.int64)
        h, w = labels.shape
        for dy, dx in _NEIGHBOURS:
            ny, nx = ys + dy, xs + dx
            inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            neighbour = np.full(ys.size, -1, dtype=np.int64)
            neighbour[inside] = labels[ny[inside], nx[inside]]
            np.add.at(counts, (np.arange(ys.size), neighbour + 1), 1)
        counts[:, 0] = 0  # unknown neighbours do not vote
        has_vote = counts.max(axis=1) > 0
        if not has_vote.any():
            break
        labels[ys[has_vote], xs[has_vote]] = counts[has_vote].argmax(axis=1) - 1
    return labels


def fill_values(values: np.ndarray, known: np.ndarray, region: np.ndarray, max_iterations: int = 64) -> np.ndarray:
    """Fill pixels of ``region`` that are not ``known`` with the mean of known 8-neighbours."""
    values = np.array(values, dtype=np.float64, copy=True)
    known = np.array(known, dtype=bool, copy=True)
    region = np.asarray(region, dtype=bool)
    for _ in range(max_iterations):
        todo = region & ~known
        if not todo.any():
            break
        total = np.zeros_like(values)
        count = np.zeros(values.shape, dtype=np.int64)
        for dy, dx in _NEIGHBOURS:
            k = _shifted(known, dy, dx, False)
            total += np.where(k, _shifted(values, dy, dx, 0.0), 0.0)
            count += k
        update = todo & (count > 0)
        if not update.any():
            break
        values[update] = total[update] / count[update]
        known |= update
    return values


def labels_to_part_masks(labels: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """``(H, W)`` label map (``-1`` = background) -> ``(24, H, W)`` float32 masks.

    Each pixel's coverage value goes to the mask of its part, so the 24 masks
    are disjoint and their sum equals ``coverage`` wherever a label is known.
    """
    labels = np.asarray(labels)
    coverage = np.asarray(coverage, dtype=np.float32)
    masks = np.zeros((NUM_PARTS,) + labels.shape, dtype=np.float32)
    ys, xs = np.nonzero(labels >= 0)
    masks[labels[ys, xs], ys, xs] = coverage[ys, xs]
    return masks
