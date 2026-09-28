"""Prepare the official SMPL / SMPL-H files for smplx, without chumpy.

The official SMPL and MANO / SMPL-H downloads are Python 2 pickles containing
``chumpy`` arrays. smplx cannot load them unless chumpy is installed, and
chumpy no longer works with current Python / NumPy releases. This script:

* reads the official files, either extracted or directly inside the official
  zip archives, without importing chumpy: chumpy objects are replaced by
  inert placeholders during unpickling and converted to NumPy arrays;
* converts sparse matrices to dense arrays, so loading needs neither chumpy
  nor scipy;
* for SMPL-H, adds the MANO hand parameters that smplx expects
  (``hands_componentsl/r`` and ``hands_meanl/r``);
* writes the smplx folder layout used by this node pack::

      <output>/smpl/SMPL_MALE.pkl     <output>/smplh/SMPLH_MALE.pkl
      <output>/smpl/SMPL_FEMALE.pkl   <output>/smplh/SMPLH_FEMALE.pkl

Only plain chumpy ``Ch`` arrays are supported, which is what the official
files contain; anything else stops the conversion with an explicit error.

Usage::

    python scripts/prepare_models.py --source D:/SMPL-downloads
    python scripts/prepare_models.py --source D:/SMPL-downloads --output D:/body_models --force
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Official file names, plus the smplx names for files that were only renamed
# (matched case-insensitively on the base name).
PATTERNS = {
    ("SMPL", "male"): re.compile(r"^(basicmodel_m_lbs_10_207_0_v1\.\d\.\d|smpl_male)\.pkl$", re.I),
    ("SMPL", "female"): re.compile(r"^(basicmodel_f_lbs_10_207_0_v1\.\d\.\d|smpl_female)\.pkl$", re.I),
    ("SMPL-H", "male"): re.compile(r"^smplh_male\.pkl$", re.I),
    ("SMPL-H", "female"): re.compile(r"^smplh_female\.pkl$", re.I),
    ("MANO", "left"): re.compile(r"^mano_left\.pkl$", re.I),
    ("MANO", "right"): re.compile(r"^mano_right\.pkl$", re.I),
}

OUTPUT_NAMES = {
    ("SMPL", "male"): ("smpl", "SMPL_MALE.pkl"),
    ("SMPL", "female"): ("smpl", "SMPL_FEMALE.pkl"),
    ("SMPL-H", "male"): ("smplh", "SMPLH_MALE.pkl"),
    ("SMPL-H", "female"): ("smplh", "SMPLH_FEMALE.pkl"),
}

EXPECTED = {
    "SMPL": {"v_template": (6890, 3), "J_regressor": (24, 6890), "weights": (6890, 24), "f": (13776, 3)},
    "SMPL-H": {"v_template": (6890, 3), "J_regressor": (52, 6890), "weights": (6890, 52), "f": (13776, 3)},
}


# ------------------------------------------------------------- unpickling


class ChumpyPlaceholder:
    """Stands in for any chumpy class while unpickling; keeps the pickled state."""

    chumpy_class = "chumpy"

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def to_array(self) -> np.ndarray:
        state = self.__dict__.get("_state")
        if self.chumpy_class != "chumpy.ch.Ch" or not isinstance(state, dict) or "x" not in state:
            raise ValueError(
                f"unsupported chumpy object {self.chumpy_class}; only plain chumpy.ch.Ch "
                "arrays can be converted without chumpy"
            )
        return np.array(to_plain(state["x"]))


class OfficialModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "chumpy" or module.startswith("chumpy."):
            return type(name, (ChumpyPlaceholder,), {"chumpy_class": f"{module}.{name}"})
        if module.startswith("numpy.core") and int(np.__version__.split(".")[0]) >= 2:
            module = "numpy._core" + module[len("numpy.core"):]
        if module.startswith("scipy.sparse."):
            # Old pickles reference scipy.sparse.csc / .coo ..., which SciPy deprecates;
            # the classes are available from the public namespace.
            sparse = importlib.import_module("scipy.sparse")
            if hasattr(sparse, name):
                return getattr(sparse, name)
        return super().find_class(module, name)


def to_plain(value):
    """Recursively replace chumpy placeholders and sparse matrices by NumPy arrays."""
    if isinstance(value, ChumpyPlaceholder):
        return value.to_array()
    if hasattr(value, "toarray") and type(value).__module__.startswith("scipy.sparse"):
        return np.asarray(value.toarray())
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_plain(v) for v in value)
    return value


def load_official(payload: bytes) -> dict:
    data = OfficialModelUnpickler(io.BytesIO(payload), encoding="latin1").load()
    if not isinstance(data, dict):
        raise ValueError(f"expected a dict, got {type(data).__name__}")
    return to_plain(data)


# ------------------------------------------------------------ discovery


def iter_candidates(source: Path) -> Iterator[Tuple[str, str, bytes]]:
    """Yield ``(display_path, base_name, payload)`` for pickles in folders and zip archives."""
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    for path in paths:
        if path.suffix.lower() == ".pkl":
            yield str(path), path.name, path.read_bytes()
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    base = Path(info.filename).name
                    if base.lower().endswith(".pkl") and not info.is_dir():
                        yield f"{path}!{info.filename}", base, archive.read(info)


def discover(source: Path) -> Dict[Tuple[str, str], Tuple[str, bytes]]:
    found: Dict[Tuple[str, str], Tuple[str, bytes]] = {}
    for display, base, payload in iter_candidates(source):
        if base.startswith("._"):  # macOS resource forks inside zips
            continue
        for key, pattern in PATTERNS.items():
            if pattern.match(base) and key not in found:
                found[key] = (display, payload)
    return found


# ----------------------------------------------------------- conversion


def merge_mano(smplh: dict, left: dict, right: dict) -> dict:
    """Add the MANO hand PCA basis and mean pose under the names smplx reads."""
    merged = dict(smplh)
    for suffix, hand in (("l", left), ("r", right)):
        merged["hands_components" + suffix] = np.asarray(hand["hands_components"])
        merged["hands_mean" + suffix] = np.asarray(hand["hands_mean"])
    return merged


def validate(model_type: str, data: dict) -> None:
    for key, shape in EXPECTED[model_type].items():
        if key not in data:
            raise ValueError(f"{model_type}: missing '{key}'")
        actual = np.shape(data[key])
        if actual != shape:
            raise ValueError(f"{model_type}: '{key}' has shape {actual}, expected {shape}")
    if np.shape(data["shapedirs"])[:2] != (6890, 3) or np.shape(data["shapedirs"])[2] < 10:
        raise ValueError(f"{model_type}: unexpected shapedirs shape {np.shape(data['shapedirs'])}")
    if model_type == "SMPL-H":
        for key in ("hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr"):
            if key not in data:
                raise ValueError(f"SMPL-H: missing '{key}' (MANO merge failed)")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="folder (searched recursively, zip archives included) or a single file")
    parser.add_argument("--output", type=Path, default=REPO / "models",
                        help="output root (default: the node pack's models/ folder)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args(argv)

    found = discover(args.source)
    for key, (display, payload) in sorted(found.items()):
        print(f"found {key[0]:6} {key[1]:6} {display}  sha256={hashlib.sha256(payload).hexdigest()}")
    if not found:
        print("No official SMPL / SMPL-H / MANO files found under", args.source)
        return 1

    written = 0
    mano = {}
    if ("MANO", "left") in found and ("MANO", "right") in found:
        mano = {side: load_official(found[("MANO", side)][1]) for side in ("left", "right")}

    for key, (subdir, name) in OUTPUT_NAMES.items():
        if key not in found:
            continue
        model_type = key[0]
        destination = args.output / subdir / name
        if destination.exists() and not args.force:
            print(f"skip   {destination} (exists; use --force to overwrite)")
            continue
        data = load_official(found[key][1])
        if model_type == "SMPL-H":
            if "hands_componentsl" not in data:
                if not mano:
                    print(f"skip   {destination}: MANO_LEFT.pkl and MANO_RIGHT.pkl are needed for SMPL-H")
                    continue
                data = merge_mano(data, mano["left"], mano["right"])
        validate(model_type, data)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(data, handle, protocol=4)
        tmp.replace(destination)
        written += 1
        print(f"wrote  {destination}")

    print(f"{written} model file(s) written.")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
