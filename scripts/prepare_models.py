"""Prepare the official SMPL / SMPL-H files for smplx, without chumpy.

The official SMPL and MANO / SMPL-H downloads are Python 2 pickles containing
``chumpy`` arrays. smplx cannot load them unless chumpy is installed, and
chumpy's PyPI release no longer works with current NumPy releases. This script:

* reads the official files, either extracted or directly inside the official
  zip archives, without importing chumpy or SciPy: chumpy objects and sparse
  matrices are replaced by inert placeholders during unpickling and converted
  to NumPy arrays;
* converts sparse matrices to dense arrays, so loading needs neither chumpy
  nor scipy;
* for SMPL-H, adds the MANO hand parameters that smplx expects
  (``hands_componentsl/r`` and ``hands_meanl/r``);
* writes the smplx folder layout used by this node pack::

      <output>/smpl/SMPL_MALE.pkl     <output>/smplh/SMPLH_MALE.pkl
      <output>/smpl/SMPL_FEMALE.pkl   <output>/smplh/SMPLH_FEMALE.pkl

  where <output> is ComfyUI/models/body_models when the pack is installed in
  ComfyUI/custom_nodes, and the pack's own models/ folder otherwise.

Only plain chumpy ``Ch`` arrays are supported, which is what the official
files contain; a file holding anything else is reported and skipped, like a
corrupt or truncated one, and the other files are still converted.
The gender-neutral SMPL model is reported and skipped: this node pack only
offers the male and female models.

Usage::

    python scripts/prepare_models.py --source D:/SMPL-downloads
    python scripts/prepare_models.py --source D:/SMPL-downloads --output D:/body_models --force
"""

from __future__ import annotations

import argparse
import io
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

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

MANO_KEYS = {"hands_components", "hands_mean"}

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


class SparsePlaceholder:
    """Stands in for a pickled scipy.sparse matrix; rebuilt densely without SciPy."""

    sparse_class = "scipy.sparse"

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def to_array(self) -> np.ndarray:
        state = self.__dict__.get("_state")
        kind = self.sparse_class.rsplit(".", 1)[-1]
        if not isinstance(state, dict) or "_shape" not in state or "data" not in state:
            raise ValueError(f"unsupported sparse matrix pickle ({self.sparse_class})")
        data = np.asarray(state["data"])
        dense = np.zeros(tuple(int(n) for n in state["_shape"]), dtype=data.dtype)
        if kind in ("csc_matrix", "csr_matrix", "csc_array", "csr_array"):
            indices = np.asarray(state["indices"], dtype=np.int64)
            indptr = np.asarray(state["indptr"], dtype=np.int64)
            major = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
            rows, cols = (indices, major) if kind.startswith("csc") else (major, indices)
        elif kind in ("coo_matrix", "coo_array"):
            if "coords" in state:  # SciPy >= 1.13 pickles a (row, col) tuple
                rows, cols = (np.asarray(c, dtype=np.int64) for c in state["coords"])
            else:
                rows = np.asarray(state["row"], dtype=np.int64)
                cols = np.asarray(state["col"], dtype=np.int64)
        else:
            raise ValueError(f"unsupported sparse matrix type {self.sparse_class}")
        np.add.at(dense, (rows, cols), data)  # duplicates are summed, as SciPy does
        return dense


class OfficialModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "chumpy" or module.startswith("chumpy."):
            return type(name, (ChumpyPlaceholder,), {"chumpy_class": f"{module}.{name}"})
        if module == "scipy.sparse" or module.startswith("scipy.sparse."):
            return type(name, (SparsePlaceholder,), {"sparse_class": f"{module}.{name}"})
        if module.startswith("numpy.core") and int(np.__version__.split(".")[0]) >= 2:
            module = "numpy._core" + module[len("numpy.core"):]
        return super().find_class(module, name)


def to_plain(value):
    """Recursively replace chumpy placeholders and sparse matrices by NumPy arrays."""
    if isinstance(value, (ChumpyPlaceholder, SparsePlaceholder)):
        return value.to_array()
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_plain(v) for v in value)
    return value


class ModelFileError(ValueError):
    """A model file could not be read; the message names the file."""


def load_official(payload: bytes, display: str = "<bytes>") -> dict:
    try:
        data = OfficialModelUnpickler(io.BytesIO(payload), encoding="latin1").load()
    except Exception as exc:  # truncated download, HTML page, wrong file ...
        raise ModelFileError(
            f"{display} could not be read as a model file ({type(exc).__name__}: {exc}); "
            "it is probably truncated or not the official pickle (re-download it)"
        ) from exc
    if not isinstance(data, dict):
        raise ModelFileError(f"{display}: expected a dict, got {type(data).__name__}")
    return to_plain(data)


# ------------------------------------------------------------ discovery


def iter_candidates(source: Path) -> Iterator[Tuple[str, str, Callable[[], bytes]]]:
    """Yield ``(display_path, base_name, read)`` for pickles in folders and zip archives.

    ``read()`` loads the bytes on demand, so only files whose name matches are
    ever read.
    """
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    for path in paths:
        if path.suffix.lower() == ".pkl":
            yield str(path), path.name, path.read_bytes
        elif path.suffix.lower() == ".zip":
            try:
                archive = zipfile.ZipFile(path)
            except zipfile.BadZipFile:
                print(f"note   {path} is not a valid zip archive; skipped")
                continue
            with archive:
                for info in archive.infolist():
                    base = Path(info.filename).name
                    if base.lower().endswith(".pkl") and not info.is_dir():
                        yield f"{path}!{info.filename}", base, (lambda p=path, n=info.filename:
                                                                 zipfile.ZipFile(p).read(n))


NEUTRAL = re.compile(r"^(basicmodel_neutral_lbs_10_207_0_v1\.\d\.\d|smpl_neutral)\.pkl$", re.I)


def version_key(display: str) -> Tuple[int, ...]:
    """Sort key preferring newer official releases (``v1.1.0`` over ``v1.0.0``).

    Only the file name counts, not a version-like folder or archive name.
    """
    base = re.split(r"[\\/!]", display)[-1]
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)", base, re.I)
    return tuple(int(x) for x in match.groups()) if match else (0, 0, 0)


Candidate = Tuple[str, Callable[[], bytes]]


def discover(source: Path) -> Dict[Tuple[str, str], List[Candidate]]:
    """Map ``(model, variant)`` to its matching files, newest official version first."""
    found: Dict[Tuple[str, str], List[Candidate]] = {}
    for display, base, read in iter_candidates(source):
        if base.startswith("._"):  # macOS resource forks inside zips
            continue
        if NEUTRAL.match(base):
            print(f"note   {display}: gender-neutral SMPL is not used by this node pack; skipped")
            continue
        for key, pattern in PATTERNS.items():
            if pattern.match(base):
                found.setdefault(key, []).append((display, read))
    for candidates in found.values():
        candidates.sort(key=lambda c: (tuple(-v for v in version_key(c[0])), c[0]))
    return found


def load_first(candidates: List[Candidate]) -> Optional[Tuple[str, dict]]:
    """Load the first readable candidate; unreadable ones are reported and skipped."""
    for display, read in candidates:
        try:
            return display, load_official(read(), display)
        except (ModelFileError, OSError, KeyError, zipfile.BadZipFile) as exc:
            print(f"error  {exc}")
        except ValueError as exc:  # unsupported chumpy objects or sparse matrices
            print(f"error  {display}: {exc}")
    return None


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


def default_output(repo: Path = REPO) -> Path:
    """ComfyUI/models/body_models when the pack sits in ComfyUI/custom_nodes, else <pack>/models."""
    comfy_models = repo.parent.parent / "models"
    if repo.parent.name.casefold() == "custom_nodes" and comfy_models.is_dir():
        return comfy_models / "body_models"
    return repo / "models"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="folder (searched recursively, zip archives included) or a single file")
    parser.add_argument("--output", type=Path, default=None,
                        help="output root (default: ComfyUI/models/body_models when the pack is "
                             "installed in ComfyUI/custom_nodes, else the pack's models/ folder)")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = default_output()
    print(f"output {args.output}")

    found = discover(args.source)
    for key, candidates in sorted(found.items()):
        for index, (display, _) in enumerate(candidates):
            status = "found " if index == 0 else "also  "
            print(f"{status} {key[0]:6} {key[1]:6} {display}" + ("" if index == 0 else "  (used only if the first one is unreadable)"))
    if not found:
        print("No official SMPL / SMPL-H / MANO files found under", args.source)
        return 1

    written = existing = skipped = 0
    mano = {}
    if ("MANO", "left") in found and ("MANO", "right") in found:
        loaded = {side: load_first(found[("MANO", side)]) for side in ("left", "right")}
        for side, item in loaded.items():
            if item is not None and not MANO_KEYS <= set(item[1]):
                print(f"error  {item[0]}: not a MANO file (missing {sorted(MANO_KEYS - set(item[1]))}); skipped")
                loaded[side] = None
        if all(loaded.values()):
            mano = {side: loaded[side][1] for side in ("left", "right")}

    for key, (subdir, name) in OUTPUT_NAMES.items():
        if key not in found:
            continue
        model_type = key[0]
        destination = args.output / subdir / name
        if destination.exists() and not args.force:
            print(f"skip   {destination} (exists; use --force to overwrite)")
            existing += 1
            continue
        loaded = load_first(found[key])
        if loaded is None:
            print(f"skip   {destination}: no readable {model_type} {key[1]} file")
            skipped += 1
            continue
        display, data = loaded
        try:
            if model_type == "SMPL-H" and "hands_componentsl" not in data:
                if not mano:
                    print(f"skip   {destination}: MANO_LEFT.pkl and MANO_RIGHT.pkl are needed for SMPL-H")
                    skipped += 1
                    continue
                data = merge_mano(data, mano["left"], mano["right"])
            validate(model_type, data)
        except ValueError as exc:  # wrong shapes or missing arrays
            print(f"error  {display}: {exc}; skipped")
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        try:
            with open(tmp, "wb") as handle:
                pickle.dump(data, handle, protocol=4)
            tmp.replace(destination)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        written += 1
        print(f"wrote  {destination}")

    print(f"{written} model file(s) written, {existing} already present, "
          f"{skipped} skipped (unreadable, invalid or incomplete; see the lines above).")
    return 0 if written or existing else 1


if __name__ == "__main__":
    sys.exit(main())
