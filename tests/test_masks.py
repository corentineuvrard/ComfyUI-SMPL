import itertools

import numpy as np
import pytest

from smpl_avatar import masks
from smpl_avatar.render import decode_labels, segmentation_palette
from smpl_avatar.segmentation import NUM_PARTS, PART_NAMES


def one_hot_parts(labels):
    """(H, W) labels -> (24, H, W) masks."""
    return masks.labels_to_part_masks(labels, np.ones(labels.shape, dtype=np.float32))


def test_union_selects_parts():
    labels = np.array([[0, 1, -1], [2, 15, 23]])
    parts = one_hot_parts(labels)
    union = masks.union_part_masks(parts, [1, 15])
    assert union.shape == (1, 2, 3)
    np.testing.assert_array_equal(union[0], [[0, 1, 0], [0, 1, 0]])


def test_union_of_all_parts_is_the_silhouette():
    rng = np.random.default_rng(0)
    labels = rng.integers(-1, NUM_PARTS, size=(16, 16))
    parts = one_hot_parts(labels)
    union = masks.union_part_masks(parts, range(NUM_PARTS))
    np.testing.assert_array_equal(union[0], (labels >= 0).astype(np.float32))
    assert parts.sum(axis=0).max() <= 1.0  # parts are disjoint


def test_union_empty_selection_and_batches():
    parts = np.ones((2 * NUM_PARTS, 4, 5), dtype=np.float32)
    parts[NUM_PARTS:] = 0.5
    empty = masks.union_part_masks(parts, [])
    assert empty.shape == (2, 4, 5) and not empty.any()
    union = masks.union_part_masks(parts, [3])
    np.testing.assert_array_equal(union[:, 0, 0], [1.0, 0.5])


@pytest.mark.parametrize("shape", [(1, 4, 4), (23, 4, 4), (4, 4), (0, 4, 4)])
def test_union_rejects_non_part_masks(shape):
    with pytest.raises(ValueError, match="part masks"):
        masks.union_part_masks(np.zeros(shape), [0])


def test_union_rejects_bad_indices():
    with pytest.raises(ValueError):
        masks.union_part_masks(np.zeros((NUM_PARTS, 2, 2)), [24])


def test_presets_use_known_parts():
    for name, parts in masks.PRESETS.items():
        masks.part_indices(parts)  # raises on unknown names
    assert masks.part_indices(masks.PRESETS["full body"]) == list(range(NUM_PARTS))
    upper = set(masks.PRESETS["upper body"])
    lower = set(masks.PRESETS["lower body"])
    assert upper.isdisjoint(lower) and upper | lower == set(PART_NAMES)
    with pytest.raises(KeyError):
        masks.part_indices(["tail"])


def test_fill_labels_uses_neighbour_majority():
    labels = np.array([
        [3, 3, 7],
        [3, -1, 7],
        [3, 3, -1],
    ])
    region = np.ones_like(labels, dtype=bool)
    region[2, 2] = False
    filled = masks.fill_labels(labels, region)
    assert filled[1, 1] == 3
    assert filled[2, 2] == -1  # outside the region stays unknown


def test_fill_labels_propagates_several_pixels():
    labels = np.full((1, 6), -1)
    labels[0, 0] = 5
    filled = masks.fill_labels(labels, np.ones((1, 6), dtype=bool))
    np.testing.assert_array_equal(filled, 5)


def test_fill_values_takes_neighbour_mean():
    values = np.array([[1.0, 0.0, 3.0]])
    known = np.array([[True, False, True]])
    filled = masks.fill_values(values, known, np.ones_like(known))
    np.testing.assert_allclose(filled, [[1.0, 2.0, 3.0]])


def test_labels_to_part_masks_keeps_coverage():
    labels = np.array([[0, 1], [-1, 1]])
    coverage = np.array([[0.25, 1.0], [0.5, 0.75]], dtype=np.float32)
    parts = masks.labels_to_part_masks(labels, coverage)
    assert parts.shape == (NUM_PARTS, 2, 2) and parts.dtype == np.float32
    np.testing.assert_allclose(parts.sum(axis=0), [[0.25, 1.0], [0.0, 0.75]])


def test_palette_blends_never_look_like_a_part():
    palette = segmentation_palette().astype(np.float64)
    assert len({tuple(c) for c in palette}) == NUM_PARTS
    exact = {tuple(c) for c in palette.astype(np.int64)}
    background = np.zeros(3)
    colors = list(palette) + [background]
    # Every 4-sample MSAA resolve that mixes at least two different sources.
    for combo in itertools.combinations_with_replacement(range(len(colors)), 4):
        if len(set(combo)) < 2:
            continue
        mean = np.mean([colors[i] for i in combo], axis=0)
        for rounding in (np.floor, np.round, np.ceil):
            assert tuple(rounding(mean).astype(np.int64)) not in exact


def test_decode_labels_exact_blend_and_background():
    palette = segmentation_palette()
    seg = np.zeros((3, 4, 3), dtype=np.uint8)
    seg[0, 0] = palette[4]
    seg[0, 1] = palette[4]
    seg[1, 0] = palette[4]
    seg[1, 1] = (palette[4].astype(int) + palette[9].astype(int)) // 2  # part/part blend
    seg[2, 3] = (palette[20].astype(int) // 4)  # isolated part/background blend
    region = np.zeros((3, 4), dtype=bool)
    region[:2, :2] = True
    region[2, 3] = True
    labels = decode_labels(seg, region)
    assert labels[0, 0] == labels[0, 1] == labels[1, 0] == 4
    assert labels[1, 1] == 4  # neighbour majority
    assert labels[2, 3] == 20  # unblended nearest palette colour
    assert (labels[~region] == -1).all()
