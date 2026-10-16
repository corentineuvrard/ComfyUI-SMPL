"""Synthetic stand-ins for SMPL / SMPL-H model files.

They reproduce the *format* smplx reads (array names and shapes, SMPL mesh
sizes: 6890 vertices, 13776 faces) with made-up content: the body is an
ellipsoid, shape direction 0 changes the height and direction 1 the girth.
No data from the real models is used. This lets the whole pipeline run in
tests without the licensed files.
"""

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPLH_PARENTS = SMPL_PARENTS[:22] + [
    p for wrist, start in ((20, 22), (21, 37)) for f in range(5)
    for p in (wrist, start + 3 * f, start + 3 * f + 1)
]
RINGS, SEGMENTS = 84, 82  # 2 + 84 * 82 = 6890 vertices, 2 * 84 * 82 = 13776 faces


def ellipsoid(radii=(0.18, 0.85, 0.12)):
    """Closed genus-0 triangulated ellipsoid with outward faces."""
    theta = np.pi * np.arange(1, RINGS + 1) / (RINGS + 1)
    phi = 2 * np.pi * np.arange(SEGMENTS) / SEGMENTS
    ring = np.stack([
        np.outer(np.sin(theta), np.cos(phi)),
        np.repeat(np.cos(theta)[:, None], SEGMENTS, axis=1),
        np.outer(np.sin(theta), np.sin(phi)),
    ], axis=-1).reshape(-1, 3)
    unit = np.vstack([[0.0, 1.0, 0.0], ring, [0.0, -1.0, 0.0]])
    vertices = unit * np.asarray(radii)

    def index(i, j):
        return 1 + i * SEGMENTS + (j % SEGMENTS)

    faces = []
    for j in range(SEGMENTS):
        faces.append((0, index(0, j), index(0, j + 1)))
        faces.append((len(vertices) - 1, index(RINGS - 1, j + 1), index(RINGS - 1, j)))
        for i in range(RINGS - 1):
            a, b = index(i, j), index(i, j + 1)
            c, d = index(i + 1, j + 1), index(i + 1, j)
            faces += [(a, d, c), (a, c, b)]
    faces = np.array(faces, dtype=np.int64)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    inward = (normals * vertices[faces].mean(axis=1)).sum(axis=1) < 0
    faces[inward] = faces[inward][:, ::-1]
    return vertices, faces


def model_dict(model_type="SMPL"):
    vertices, faces = ellipsoid()
    num_joints = 24 if model_type == "SMPL" else 52
    parents = SMPL_PARENTS if model_type == "SMPL" else SMPLH_PARENTS
    shapedirs = np.zeros(vertices.shape + (10,))
    shapedirs[:, 1, 0] = vertices[:, 1] * 0.06
    shapedirs[:, 0, 1] = vertices[:, 0] * 0.10
    shapedirs[:, 2, 1] = vertices[:, 2] * 0.10
    for k in range(2, 10):
        shapedirs[:, 0, k] = 0.004 * vertices[:, 0] * np.sin(k * vertices[:, 1])
    j_regressor = np.zeros((num_joints, len(vertices)))
    j_regressor[:, [0, -1]] = 0.5  # every joint at the centre
    weights = np.zeros((len(vertices), num_joints))
    weights[:, 0] = 1.0
    kintree = np.array([np.array(parents) % 2**32, np.arange(num_joints)], dtype=np.uint32)
    data = {
        "v_template": vertices,
        "shapedirs": shapedirs,
        # All-zero pose correctives, stored as float32 to keep the files small.
        "posedirs": np.zeros(vertices.shape + (9 * (num_joints - 1),), dtype=np.float32),
        "J_regressor": j_regressor,
        "weights": weights,
        "kintree_table": kintree,
        "f": faces.astype(np.uint32),
        "bs_style": "lbs",
        "bs_type": "lrotmin",
    }
    return data


def mano_dict():
    return {"hands_components": np.eye(45), "hands_mean": np.zeros(45), "hands_coeffs": np.zeros((10, 45))}


def write_clean_models(root: Path):
    """smplx layout with plain NumPy arrays (what prepare_models.py produces)."""
    smplh = model_dict("SMPL-H")
    smplh.update(hands_componentsl=np.eye(45), hands_componentsr=np.eye(45),
                 hands_meanl=np.zeros(45), hands_meanr=np.zeros(45))
    for folder, name, data in (
        ("smpl", "SMPL_{}.pkl", model_dict("SMPL")),
        ("smplh", "SMPLH_{}.pkl", smplh),
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for gender in ("MALE", "FEMALE"):
            with open(root / folder / name.format(gender), "wb") as handle:
                pickle.dump(data, handle, protocol=4)


class _FakeCh:
    """Mimics how chumpy pickles a Ch array: its __dict__ with the value in 'x'."""

    def __init__(self, x):
        self.x = np.asarray(x)
        self._dirty_vars = set()


class _FakeCsc:
    """Mimics how SciPy pickles a csc_matrix: its __dict__ (indptr / indices / data / _shape)."""

    def __init__(self, dense):
        dense = np.asarray(dense)
        indptr, indices, data = [0], [], []
        for col in range(dense.shape[1]):
            rows = np.flatnonzero(dense[:, col])
            indices.extend(int(r) for r in rows)
            data.extend(dense[rows, col])
            indptr.append(len(indices))
        self.format = "csc"
        self._shape = tuple(dense.shape)
        self.indptr = np.array(indptr, dtype=np.int32)
        self.indices = np.array(indices, dtype=np.int32)
        self.data = np.array(data, dtype=dense.dtype)
        self.maxprint = 50


def dumps_with_fake_chumpy(data: dict, chumpy_keys=("shapedirs", "v_template", "posedirs"),
                           sparse_keys=()) -> bytes:
    """Pickle ``data`` as the official files do: some arrays wrapped in chumpy.ch.Ch,
    and ``sparse_keys`` stored as ``scipy.sparse.csc.csc_matrix`` objects."""
    fake_pkg = types.ModuleType("chumpy")
    fake_mod = types.ModuleType("chumpy.ch")
    cls = type("Ch", (_FakeCh,), {"__module__": "chumpy.ch"})
    fake_mod.Ch = cls
    fake_pkg.ch = fake_mod
    csc = type("csc_matrix", (_FakeCsc,), {"__module__": "scipy.sparse.csc"})
    fake_scipy = types.ModuleType("scipy")
    fake_sparse = types.ModuleType("scipy.sparse")
    fake_csc = types.ModuleType("scipy.sparse.csc")
    fake_csc.csc_matrix = csc
    fake_sparse.csc = fake_csc
    fake_scipy.sparse = fake_sparse
    fakes = {"chumpy": fake_pkg, "chumpy.ch": fake_mod,
             "scipy": fake_scipy, "scipy.sparse": fake_sparse, "scipy.sparse.csc": fake_csc}
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in fakes}  # None is a valid entry
    sys.modules.update(fakes)
    try:
        wrapped = {k: (cls(v) if k in chumpy_keys else csc(v) if k in sparse_keys else v)
                   for k, v in data.items()}
        return pickle.dumps(wrapped, protocol=2)
    finally:
        for name, module in saved.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def write_segmentation(path: Path, vertices=None):
    """24 parts made of horizontal bands of the ellipsoid (boundary rows shared)."""
    from smpl_avatar.segmentation import PART_NAMES

    if vertices is None:
        vertices, _ = ellipsoid()
    order = np.argsort(-vertices[:, 1], kind="stable")
    bands = np.array_split(order, len(PART_NAMES))
    data = {name: sorted(int(i) for i in band) for name, band in zip(PART_NAMES, bands)}
    data["head"] = sorted(set(data["head"]) | {int(bands[1][0])})  # one shared vertex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path
