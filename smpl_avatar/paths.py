"""Where body model files, the regressor and the segmentation live."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_MODELS_DIR = PACK_DIR / "models"

# Extra body-model roots, separated by os.pathsep. Each root uses the smplx
# layout: <root>/smpl/SMPL_MALE.pkl, <root>/smplh/SMPLH_FEMALE.pkl, ...
ENV_MODEL_DIR = "SMPL_AVATAR_MODEL_DIR"

REGRESSOR_PATH = PACK_MODELS_DIR / "regressor" / "height_weight_to_betas.npz"
SEGMENTATION_PATH = PACK_MODELS_DIR / "segmentation" / "smpl_vert_segmentation.json"

# model type -> (accepted sub-folder names, accepted file names; {G} is MALE or
# FEMALE). Names are compared case-insensitively; the first ones are canonical.
MODEL_FILES = {
    "SMPL": (("smpl",), ("SMPL_{G}.pkl",)),
    "SMPL-H": (("smplh", "smpl-h", "smpl_h"), ("SMPLH_{G}.pkl", "SMPLH_{G}.npz")),
}


class BodyModelNotFoundError(FileNotFoundError):
    """Raised when no body model file matches the requested type and gender."""


def body_model_roots() -> List[Path]:
    """Search order: $SMPL_AVATAR_MODEL_DIR, <ComfyUI>/models/body_models, <pack>/models."""
    roots: List[Path] = []
    for entry in os.environ.get(ENV_MODEL_DIR, "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry.strip()).expanduser())
    try:
        import folder_paths  # provided by ComfyUI

        roots.append(Path(folder_paths.models_dir) / "body_models")
    except Exception:  # not running inside ComfyUI
        pass
    roots.append(PACK_MODELS_DIR)
    unique: List[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _matching_entries(folder: Path, names: Sequence[str], want_dir: bool) -> List[Path]:
    """Entries of ``folder`` named like one of ``names`` (case-insensitive), in ``names`` order."""
    rank = {name.casefold(): i for i, name in enumerate(names)}
    try:
        entries = [e for e in folder.iterdir() if e.name.casefold() in rank]
    except OSError:
        return []
    entries = [e for e in entries if (e.is_dir() if want_dir else e.is_file())]
    return sorted(entries, key=lambda e: (rank[e.name.casefold()], e.name))


def _spec(model_type: str, gender: str):
    if model_type not in MODEL_FILES:
        raise ValueError(f"unknown model type {model_type!r}; expected one of {list(MODEL_FILES)}")
    subdirs, patterns = MODEL_FILES[model_type]
    return subdirs, [pattern.format(G=gender.upper()) for pattern in patterns]


def candidate_files(model_type: str, gender: str) -> List[Path]:
    """Canonical expected locations, in search order (for error messages)."""
    subdirs, names = _spec(model_type, gender)
    return [root / subdirs[0] / name for root in body_model_roots() for name in names]


def find_body_model(model_type: str, gender: str) -> Path:
    subdirs, names = _spec(model_type, gender)
    for root in body_model_roots():
        for folder in _matching_entries(root, subdirs, want_dir=True):
            files = _matching_entries(folder, names, want_dir=False)
            if files:
                return files[0]
    searched = "\n  ".join(str(p) for p in candidate_files(model_type, gender))
    raise BodyModelNotFoundError(
        f"No {model_type} {gender} body model found. Looked for:\n  {searched}\n"
        "(folder and file names are matched case-insensitively; SMPL-H also accepts "
        "'smpl-h' / 'smpl_h' folders). Body model files are not distributed with this node "
        "pack: download them from the official website and prepare them as described in the "
        "README (section 'Body model files')."
    )
