"""Linear regressor from (height_cm, weight_kg) to SMPL shape coefficients.

The regressor is an affine map ``betas = [height_cm, weight_kg] @ weights + bias``
stored as plain NumPy arrays in an ``.npz`` file (no pickle, no scikit-learn).

It is produced by a fully synthetic procedure that only needs the body model the
user downloaded themselves:

1. Sample shape vectors ``betas ~ N(0, beta_std**2 * I)``.
2. Build the rest-pose mesh ``v_template + shapedirs @ betas`` for each sample.
3. Measure height (vertex extent along the vertical axis) and weight
   (enclosed mesh volume times a constant body density).
4. Fit ordinary least squares ``[height, weight, 1] -> betas``.
5. Report the round-trip error: rebuild the mesh from the predicted betas and
   compare its measured height and weight with the requested ones, on a grid
   of heights x BMI values and on held-out samples.

Everything in this module is NumPy only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

FORMAT_VERSION = 1
FEATURE_NAMES = ("height_cm", "weight_kg")
NUM_BETAS = 10
BODY_DENSITY_KG_M3 = 985.0
VERTICAL_AXIS = 1  # SMPL meshes are y-up

DEFAULT_SAMPLES = 20000
DEFAULT_SEED = 0
DEFAULT_BETA_STD = 1.0


@dataclass(frozen=True)
class LinearBetaRegressor:
    """Affine map from ``(height_cm, weight_kg)`` to ``NUM_BETAS`` betas."""

    weights: np.ndarray  # (2, NUM_BETAS)
    bias: np.ndarray  # (NUM_BETAS,)
    meta: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] != len(FEATURE_NAMES):
            raise ValueError(f"weights must have shape (2, n), got {weights.shape}")
        if bias.shape != (weights.shape[1],):
            raise ValueError(f"bias must have shape ({weights.shape[1]},), got {bias.shape}")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)

    @property
    def num_betas(self) -> int:
        return self.weights.shape[1]

    def predict(self, height_cm, weight_kg) -> np.ndarray:
        """Return betas of shape ``(NUM_BETAS,)`` for scalars, ``(N, NUM_BETAS)`` for arrays."""
        h = np.asarray(height_cm, dtype=np.float64)
        w = np.asarray(weight_kg, dtype=np.float64)
        features = np.stack(np.broadcast_arrays(h, w), axis=-1)
        return features @ self.weights + self.bias

    def affine_matrix(self) -> np.ndarray:
        """Return the ``(3, NUM_BETAS)`` matrix acting on ``[height_cm, weight_kg, 1]``."""
        return np.vstack([self.weights, self.bias[None, :]])


def regressor_key(model_type: str, gender: str) -> str:
    """Storage key, e.g. ``("SMPL-H", "female") -> "smplh_female"``."""
    return f"{model_type.lower().replace('-', '')}_{gender.lower()}"


# --------------------------------------------------------------------------- I/O


def save_regressors(path, regressors: Mapping[str, LinearBetaRegressor]) -> None:
    """Write regressors to ``path`` (``.npz``) atomically."""
    path = Path(path)
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.array(FORMAT_VERSION),
        "feature_names": np.array(FEATURE_NAMES),
        "keys": np.array(sorted(regressors)),
    }
    for key in sorted(regressors):  # fixed order: identical input gives identical bytes
        reg = regressors[key]
        if "__" in key:
            raise ValueError(f"invalid regressor key {key!r}")
        arrays[f"{key}__weights"] = reg.weights
        arrays[f"{key}__bias"] = reg.bias
        arrays[f"{key}__meta"] = np.array(json.dumps(dict(reg.meta), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    tmp.replace(path)


def load_regressors(path) -> Dict[str, LinearBetaRegressor]:
    """Read regressors written by :func:`save_regressors` (never unpickles)."""
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported regressor format version {version}")
        features = tuple(str(name) for name in data["feature_names"])
        if features != FEATURE_NAMES:
            raise ValueError(f"unexpected feature names {features}")
        result = {}
        for key in (str(k) for k in data["keys"]):
            result[key] = LinearBetaRegressor(
                weights=data[f"{key}__weights"],
                bias=data[f"{key}__bias"],
                meta=json.loads(str(data[f"{key}__meta"])),
            )
        return result


# ------------------------------------------------------------------ measurement


def shape_basis(v_template: np.ndarray, shapedirs: np.ndarray, num_betas: int = NUM_BETAS):
    """Return ``(template_flat (3V,), basis (3V, num_betas))`` in float64."""
    v_template = np.asarray(v_template, dtype=np.float64)
    shapedirs = np.asarray(shapedirs, dtype=np.float64)
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != v_template.shape:
        raise ValueError(
            f"shapedirs {shapedirs.shape} does not match v_template {v_template.shape}"
        )
    if shapedirs.shape[2] < num_betas:
        raise ValueError(f"model has only {shapedirs.shape[2]} shape directions")
    basis = shapedirs[:, :, :num_betas].reshape(-1, num_betas)
    return v_template.reshape(-1), basis


def mesh_heights(vertices: np.ndarray) -> np.ndarray:
    """Vertex extent along the vertical axis; ``vertices`` is ``(..., V, 3)``."""
    y = np.asarray(vertices)[..., VERTICAL_AXIS]
    return y.max(axis=-1) - y.min(axis=-1)


def mesh_volumes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Enclosed volume of closed triangle meshes (divergence theorem).

    ``vertices`` is ``(..., V, 3)``; the absolute value is returned so the
    result does not depend on the face winding convention.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    a = vertices[..., faces[:, 0], :]
    b = vertices[..., faces[:, 1], :]
    c = vertices[..., faces[:, 2], :]
    signed = np.einsum("...fi,...fi->...", a, np.cross(b, c)) / 6.0
    return np.abs(signed)


class VolumePolynomial:
    """Exact enclosed volume of ``v_template + shapedirs @ betas`` as a cubic polynomial.

    Each face contributes ``a . (b x c) / 6`` where every corner is affine in
    ``betas`` (``a = a0 + A @ betas``, ...), so the total volume is a polynomial
    of degree 3 in ``betas``. Its coefficients are summed over the faces once;
    evaluating the volume of many shapes is then independent of the mesh size.
    """

    def __init__(self, v_template: np.ndarray, shapedirs: np.ndarray, faces: np.ndarray, num_betas: int = NUM_BETAS):
        template = np.asarray(v_template, dtype=np.float64)
        dirs = np.asarray(shapedirs, dtype=np.float64)[:, :, :num_betas]
        faces = np.asarray(faces, dtype=np.int64)
        a0, b0, c0 = (template[faces[:, k]] for k in range(3))  # (F, 3)
        A, B, C = (dirs[faces[:, k]] for k in range(3))  # (F, 3, K)

        b0_c0 = np.cross(b0, c0)  # (F, 3)
        B_c0 = np.cross(B, c0[:, :, None], axis=1)  # (F, 3, K): B_i x c0
        b0_C = np.cross(b0[:, :, None], C, axis=1)  # (F, 3, K): b0 x C_i
        B_C = np.cross(B[:, :, :, None], C[:, :, None, :], axis=1)  # (F, 3, K, K): B_j x C_k

        self.c0 = np.einsum("fp,fp->", a0, b0_c0)
        self.c1 = (
            np.einsum("fpi,fp->i", A, b0_c0)
            + np.einsum("fp,fpi->i", a0, B_c0)
            + np.einsum("fp,fpi->i", a0, b0_C)
        )
        self.c2 = (
            np.einsum("fpi,fpj->ij", A, B_c0)
            + np.einsum("fpi,fpj->ij", A, b0_C)
            + np.einsum("fp,fpij->ij", a0, B_C)
        )
        self.c3 = np.einsum("fpi,fpjk->ijk", A, B_C)

    def __call__(self, betas: np.ndarray) -> np.ndarray:
        """Absolute volume for each row of ``betas`` (``(N, K)`` -> ``(N,)``)."""
        b = np.atleast_2d(np.asarray(betas, dtype=np.float64))
        signed = (
            self.c0
            + b @ self.c1
            + np.einsum("ni,ij,nj->n", b, self.c2, b)
            + np.einsum("ijk,ni,nj,nk->n", self.c3, b, b, b, optimize=True)
        ) / 6.0
        return np.abs(signed)


class ShapeMeasurer:
    """Height (cm) and weight (kg) of rest-pose shapes, for many betas at once."""

    def __init__(self, v_template, shapedirs, faces, density: float = BODY_DENSITY_KG_M3):
        template, basis = shape_basis(v_template, shapedirs)
        num_vertices = np.asarray(v_template).shape[0]
        # Only the vertical coordinate is needed for the height.
        self._y0 = template.reshape(num_vertices, 3)[:, VERTICAL_AXIS]
        self._y_dirs = basis.reshape(num_vertices, 3, -1)[:, VERTICAL_AXIS, :]
        self._volume = VolumePolynomial(v_template, shapedirs, faces)
        self.density = density

    def __call__(self, betas: np.ndarray, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        betas = np.atleast_2d(np.asarray(betas, dtype=np.float64))
        heights = np.empty(len(betas))
        for start in range(0, len(betas), batch_size):
            y = self._y0[None, :] + betas[start:start + batch_size] @ self._y_dirs.T
            heights[start:start + batch_size] = (y.max(axis=1) - y.min(axis=1)) * 100.0
        return heights, self._volume(betas) * self.density


def measure_betas(
    v_template: np.ndarray,
    shapedirs: np.ndarray,
    faces: np.ndarray,
    betas: np.ndarray,
    density: float = BODY_DENSITY_KG_M3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(height_cm, weight_kg)`` of rest-pose meshes for each row of ``betas``."""
    return ShapeMeasurer(v_template, shapedirs, faces, density)(betas)


# ---------------------------------------------------------------------- fitting


def fit_least_squares(height_cm, weight_kg, betas) -> LinearBetaRegressor:
    """Ordinary least squares ``[height_cm, weight_kg, 1] -> betas``."""
    height_cm = np.asarray(height_cm, dtype=np.float64)
    weight_kg = np.asarray(weight_kg, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    design = np.column_stack([height_cm, weight_kg, np.ones_like(height_cm)])
    solution, *_ = np.linalg.lstsq(design, betas, rcond=None)
    return LinearBetaRegressor(weights=solution[:2], bias=solution[2])


def target_grid(
    heights_cm: Iterable[float], bmis: Iterable[float]
) -> np.ndarray:
    """Cartesian grid of ``(height_cm, weight_kg)`` from heights and BMI values."""
    rows = [(h, bmi * (h / 100.0) ** 2) for h in heights_cm for bmi in bmis]
    return np.asarray(rows, dtype=np.float64)


DEFAULT_GRID = {
    "male": dict(heights_cm=np.arange(155.0, 200.1, 5.0), bmis=np.arange(18.0, 35.1, 1.0)),
    "female": dict(heights_cm=np.arange(145.0, 190.1, 5.0), bmis=np.arange(17.0, 35.1, 1.0)),
}


def model_fingerprint(v_template: np.ndarray, shapedirs: np.ndarray, faces: np.ndarray) -> str:
    """SHA-256 of the arrays the regressor depends on (first NUM_BETAS directions)."""
    digest = hashlib.sha256()
    for array in (
        np.asarray(v_template, dtype=np.float32),
        np.asarray(shapedirs, dtype=np.float32)[:, :, :NUM_BETAS],
        np.asarray(faces, dtype=np.int64),
    ):
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _errors(target_h, target_w, h, w) -> Dict[str, float]:
    dh = np.abs(h - target_h)
    dw = np.abs(w - target_w)
    return {
        "height_mae_cm": float(dh.mean()),
        "height_max_cm": float(dh.max()),
        "weight_mae_kg": float(dw.mean()),
        "weight_max_kg": float(dw.max()),
        "weight_mape_pct": float((dw / target_w).mean() * 100.0),
    }


def train(
    v_template: np.ndarray,
    shapedirs: np.ndarray,
    faces: np.ndarray,
    *,
    gender: str = "male",
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    beta_std: float = DEFAULT_BETA_STD,
    density: float = BODY_DENSITY_KG_M3,
    grid: Optional[np.ndarray] = None,
    extra_meta: Optional[Mapping] = None,
) -> LinearBetaRegressor:
    """Train a regressor for one body model and return it with a fit report in ``meta``."""
    rng = np.random.default_rng(seed)
    measure = ShapeMeasurer(v_template, shapedirs, faces, density)

    betas = rng.standard_normal((samples, NUM_BETAS)) * beta_std
    heights, weights = measure(betas)
    n_train = int(samples * 0.8)
    train_slice, test_slice = slice(0, n_train), slice(n_train, samples)

    reg = fit_least_squares(heights[train_slice], weights[train_slice], betas[train_slice])
    residual = reg.predict(heights[test_slice], weights[test_slice]) - betas[test_slice]
    beta_r2 = 1.0 - (residual ** 2).sum(axis=0) / (
        (betas[test_slice] - betas[test_slice].mean(axis=0)) ** 2
    ).sum(axis=0)

    if grid is None:
        grid = target_grid(**DEFAULT_GRID.get(gender, DEFAULT_GRID["male"]))
    bmi = grid[:, 1] / (grid[:, 0] / 100.0) ** 2
    common = grid[(bmi >= 18.5) & (bmi <= 30.0)]
    # Held-out samples come from the same prior as the training set (realistic
    # bodies) but were not used for fitting.
    held_out = np.column_stack([heights[test_slice], weights[test_slice]])

    def roundtrip(points: np.ndarray) -> Dict[str, float]:
        h, w = measure(reg.predict(points[:, 0], points[:, 1]))
        return _errors(points[:, 0], points[:, 1], h, w)

    report = {
        "grid": roundtrip(grid),
        "grid_bmi_18.5_30": roundtrip(common) if len(common) else None,
        "held_out": roundtrip(held_out),
        "beta_r2_held_out": [float(x) for x in beta_r2],
    }

    meta = {
        "method": "synthetic ordinary least squares",
        "gender": gender,
        "samples": samples,
        "seed": seed,
        "beta_std": beta_std,
        "density_kg_m3": density,
        "height_definition": "vertex extent along y, rest pose",
        "weight_definition": "enclosed mesh volume * density",
        "grid_height_range_cm": [float(grid[:, 0].min()), float(grid[:, 0].max())],
        "grid_weight_range_kg": [float(grid[:, 1].min()), float(grid[:, 1].max())],
        "training_height_range_cm": [float(heights.min()), float(heights.max())],
        "training_weight_range_kg": [float(weights.min()), float(weights.max())],
        "model_fingerprint": model_fingerprint(v_template, shapedirs, faces),
        "report": report,
    }
    if extra_meta:
        meta.update(extra_meta)
    return LinearBetaRegressor(weights=reg.weights, bias=reg.bias, meta=meta)
