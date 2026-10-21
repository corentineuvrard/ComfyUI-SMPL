"""Where body model files, the regressor and the segmentation live."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

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


# Empty files created in ComfyUI/models/body_models/<folder>, like the
# put_checkpoints_here files of ComfyUI's own model folders.
PLACEHOLDERS = {
    "smpl": "put_SMPL_MALE_and_SMPL_FEMALE_here",
    "smplh": "put_SMPLH_MALE_and_SMPLH_FEMALE_here",
}


class BodyModelNotFoundError(FileNotFoundError):
    """Raised when no body model file matches the requested type and gender."""


def env_model_dirs() -> List[Path]:
    """Folders listed in $SMPL_AVATAR_MODEL_DIR (quotes around entries are tolerated)."""
    dirs: List[Path] = []
    for entry in os.environ.get(ENV_MODEL_DIR, "").split(os.pathsep):
        entry = entry.strip().strip('"').strip("'").strip()
        if entry:
            dirs.append(Path(entry).expanduser())
    return dirs


def comfyui_model_dirs() -> List[Path]:
    """ComfyUI's ``body_models`` folders: ``<models_dir>/body_models`` plus any
    ``body_models:`` entries of ``extra_model_paths.yaml`` (empty outside ComfyUI)."""
    try:
        import folder_paths  # provided by ComfyUI
    except Exception:  # not running inside ComfyUI
        return []
    default = Path(folder_paths.models_dir) / "body_models"
    dirs: List[Path] = []
    try:
        if "body_models" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("body_models", str(default))
        dirs.extend(Path(p) for p in folder_paths.get_folder_paths("body_models"))
    except Exception:  # older ComfyUI without these helpers
        pass
    dirs.append(default)
    return dirs


def create_placeholders() -> Optional[Path]:
    """Create ``<models_dir>/body_models/smpl`` and ``smplh`` with a placeholder file
    naming what goes there, so users see where the model files belong.

    Inside ComfyUI only (``None`` elsewhere). A folder that already exists is
    left untouched, so a placeholder the user deleted does not come back.
    """
    try:
        import folder_paths  # provided by ComfyUI
    except Exception:  # not running inside ComfyUI
        return None
    root = Path(folder_paths.models_dir) / "body_models"
    for folder_name, placeholder in PLACEHOLDERS.items():
        folder = root / folder_name
        if folder.exists():
            continue
        try:
            folder.mkdir(parents=True)
            (folder / placeholder).touch()
        except OSError:  # read-only installation: nothing to show, nothing to break
            return None
    return root


def body_model_roots() -> List[Path]:
    """Search order: $SMPL_AVATAR_MODEL_DIR, ComfyUI's body_models folders, <pack>/models."""
    roots: List[Path] = env_model_dirs() + comfyui_model_dirs() + [PACK_MODELS_DIR]
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
    site = "https://smpl.is.tue.mpg.de" if model_type == "SMPL" else "https://mano.is.tue.mpg.de"
    raise BodyModelNotFoundError(
        f"No {model_type} {gender} body model found. Looked for:\n  {searched}\n"
        "(folder and file names are matched case-insensitively; SMPL-H also accepts "
        "'smpl-h' / 'smpl_h' folders). Body model files are not distributed with this node "
        f"pack: download them from {site} (registration required), prepare them with "
        "'python scripts/prepare_models.py', and place them in one of the folders above or in "
        f"a folder listed in the {ENV_MODEL_DIR} environment variable (see README, "
        "'Body model files')."
    )
