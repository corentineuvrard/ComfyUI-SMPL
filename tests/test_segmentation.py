import hashlib
import json

import numpy as np
import pytest

from smpl_avatar import paths, segmentation
from smpl_avatar.segmentation import NUM_PARTS, PART_DESCRIPTIONS, PART_NAMES


def test_part_names():
    assert len(PART_NAMES) == NUM_PARTS == 24
    assert len(set(PART_NAMES)) == 24
    assert set(PART_DESCRIPTIONS) == set(PART_NAMES)
    assert PART_NAMES[0] == "hips" and PART_NAMES[15] == "head"


def small_segmentation(num_vertices=30):
    """Every part gets one vertex, the rest go to 'head'; vertex 0 is shared."""
    data = {name: [i] for i, name in enumerate(PART_NAMES)}
    data["head"] = [15] + list(range(24, num_vertices)) + [0]
    return data


def test_parse_segmentation_builds_membership():
    membership = segmentation.parse_segmentation(small_segmentation(), num_vertices=30)
    assert membership.shape == (30, 24) and membership.dtype == bool
    assert membership[0, 0] and membership[0, 15]  # shared boundary vertex
    assert membership.any(axis=1).all()


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.pop("head"), "missing"),
    (lambda d: d.update(tail=[1]), "unexpected"),
    (lambda d: d.update(head=[99]), "outside"),
    (lambda d: d.update(head=[0.5]), "integers"),
    (lambda d: d.update(head=[15]), "no part"),
])
def test_parse_segmentation_rejects_bad_files(mutate, message):
    data = small_segmentation()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        segmentation.parse_segmentation(data, num_vertices=30)


def test_face_labels_majority_then_exclusive_then_index():
    membership = np.zeros((6, NUM_PARTS), dtype=bool)
    membership[[0, 1], 2] = True  # vertices 0, 1 only in part 2
    membership[2, [2, 5]] = True  # vertex 2 shared by parts 2 and 5
    membership[[3, 4], 5] = True
    membership[5, [5, 7]] = True
    faces = np.array([
        [0, 1, 2],  # part 2 has 3 votes
        [2, 3, 4],  # part 5 has 3 votes
        [2, 5, 0],  # part 2: 2 votes, part 5: 2 votes, part 7: 1 -> exclusive vote for 2
        [2, 5, 2],  # part 5: 3 votes (vertex 2 twice + vertex 5), part 2: 2 votes
    ])
    np.testing.assert_array_equal(segmentation.face_part_labels(membership, faces), [2, 5, 2, 5])


def test_ensure_file_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv(segmentation.ENV_NO_DOWNLOAD, "1")
    with pytest.raises(segmentation.SegmentationUnavailableError, match="disabled"):
        segmentation.ensure_file(tmp_path / "missing.json")
    present = tmp_path / "present.json"
    present.write_text("{}")
    assert segmentation.ensure_file(present) == present


def test_download_verifies_checksum(tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(segmentation.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"{}"))
    with pytest.raises(segmentation.SegmentationUnavailableError, match="SHA-256"):
        segmentation.download(tmp_path / "seg.json")
    assert not (tmp_path / "seg.json").exists()


def test_url_is_pinned_to_a_commit():
    assert segmentation.SEGMENTATION_COMMIT in segmentation.SEGMENTATION_URL
    assert "/main/" not in segmentation.SEGMENTATION_URL


@pytest.mark.skipif(not paths.SEGMENTATION_PATH.is_file(), reason="segmentation file not downloaded")
def test_installed_segmentation_file_is_the_pinned_one():
    payload = paths.SEGMENTATION_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == segmentation.SEGMENTATION_SHA256
    membership = segmentation.parse_segmentation(json.loads(payload))
    assert membership.shape == (6890, 24)
