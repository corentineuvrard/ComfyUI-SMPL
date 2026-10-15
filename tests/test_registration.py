import subprocess
import sys
import textwrap

import pytest

from helpers import REPO

STANDARD_TYPES = {"IMAGE", "MASK", "INT", "FLOAT", "BOOLEAN", "STRING"}


def test_mappings(pack):
    assert set(pack.NODE_CLASS_MAPPINGS) == {"SMPLAvatarGenerate", "SMPLAvatarPartMaskUnion"}
    assert pack.NODE_DISPLAY_NAME_MAPPINGS == {
        "SMPLAvatarGenerate": "Generate SMPL Avatar",
        "SMPLAvatarPartMaskUnion": "Union SMPL Part Masks",
    }
    assert set(pack.__all__) == {"NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"}


@pytest.mark.parametrize("key", ["SMPLAvatarGenerate", "SMPLAvatarPartMaskUnion"])
def test_node_contract(pack, key):
    cls = pack.NODE_CLASS_MAPPINGS[key]
    assert callable(getattr(cls(), cls.FUNCTION))
    assert cls.CATEGORY == "SMPL Avatar"
    assert cls.DESCRIPTION
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) == len(cls.OUTPUT_TOOLTIPS)
    # Only standard ComfyUI types: no undocumented custom socket types.
    assert set(cls.RETURN_TYPES) <= {"IMAGE", "MASK"}

    spec = cls.INPUT_TYPES()
    assert set(spec) <= {"required", "optional", "hidden"}
    for section in ("required", "optional"):
        for name, (kind, options) in spec.get(section, {}).items():
            if isinstance(kind, list):
                assert options.get("default", kind[0]) in kind, name
                continue
            assert kind in STANDARD_TYPES, (name, kind)
            if kind in ("INT", "FLOAT"):
                assert options["min"] <= options["default"] <= options["max"], name
            if kind == "BOOLEAN":
                assert isinstance(options["default"], bool)


def test_generate_inputs(pack):
    spec = pack.NODE_CLASS_MAPPINGS["SMPLAvatarGenerate"].INPUT_TYPES()
    required = spec["required"]
    assert list(required) == ["model_type", "gender", "height_cm", "weight_kg", "width", "height"]
    assert required["model_type"][0] == ["SMPL", "SMPL-H"]
    assert required["gender"][0] == ["male", "female"]
    assert pack.NODE_CLASS_MAPPINGS["SMPLAvatarGenerate"].RETURN_TYPES == ("IMAGE", "MASK", "IMAGE", "MASK")


def test_union_node_inputs_list_every_part(pack):
    from smpl_avatar.segmentation import PART_NAMES

    spec = pack.NODE_CLASS_MAPPINGS["SMPLAvatarPartMaskUnion"].INPUT_TYPES()
    toggles = [name for name, (kind, _) in spec["required"].items() if kind == "BOOLEAN"]
    assert toggles == list(PART_NAMES)
    assert spec["required"]["part_masks"][0] == "MASK"


def test_union_node_execution(pack):
    torch = pytest.importorskip("torch")
    node = pack.NODE_CLASS_MAPPINGS["SMPLAvatarPartMaskUnion"]()
    parts = torch.zeros(24, 9, 9)
    parts[15, 4, 4] = 1.0  # head
    parts[12, 0, 0] = 1.0  # neck
    parts[20, 8, 8] = 1.0  # leftHand

    (mask,) = node.union(parts, preset="head", leftHand=True)
    assert mask.shape == (1, 9, 9) and mask.dtype == torch.float32
    assert mask[0, 4, 4] == 1 and mask[0, 8, 8] == 1 and mask[0, 0, 0] == 0
    assert mask.sum() == 2

    (grown,) = node.union(parts, preset="head", grow_px=1)
    assert grown.sum() == 9
    (shrunk,) = node.union(parts, preset="full body", grow_px=-1)
    assert shrunk.sum() == 0

    # Erosion treats the outside of the image as background: a full head mask
    # loses its one-pixel border strip.
    full = torch.zeros(24, 9, 9)
    full[15] = 1.0
    (eroded,) = node.union(full, preset="head", grow_px=-1)
    assert eroded.sum() == 49 and eroded[0, 0].sum() == 0 and eroded[0, :, 0].sum() == 0

    with pytest.raises(ValueError, match="part masks"):
        node.union(torch.zeros(1, 9, 9), preset="head")


def test_importing_the_pack_is_lightweight():
    """Heavy or optional dependencies must only be imported when a node runs."""
    code = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "pack", {str(REPO / "__init__.py")!r}, submodule_search_locations=[{str(REPO)!r}])
        module = importlib.util.module_from_spec(spec)
        sys.modules["pack"] = module
        spec.loader.exec_module(module)
        heavy = [m for m in ("pyrender", "OpenGL", "smplx", "trimesh", "cv2", "sklearn", "torch")
                 if m in sys.modules]
        print(",".join(heavy))
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == ""
