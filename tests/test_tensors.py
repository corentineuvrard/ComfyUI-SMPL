"""Helpers that turn NumPy results into ComfyUI tensors (nodes/tensors.py)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tensors(pack):
    return pack.nodes.tensors


def test_to_tensor_gives_a_contiguous_float32_tensor(tensors):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]  # float64 and not contiguous
    tensor = tensors.to_tensor(array)
    assert tensor.dtype == torch.float32 and tensor.is_contiguous()
    np.testing.assert_array_equal(tensor.numpy(), array)


def test_grow_mask_dilates_and_erodes_with_a_square_kernel(tensors):
    mask = torch.zeros(2, 7, 7)
    mask[0, 3, 3] = 1.0
    mask[1, 1:6, 1:6] = 1.0
    assert tensors.grow_mask(mask, 0) is mask
    grown = tensors.grow_mask(mask, 1)
    assert grown.shape == mask.shape
    assert grown[0].sum() == 9 and grown[0, 2:5, 2:5].all()
    assert grown[1].sum() == 49  # the 5 x 5 square grows to the whole 7 x 7 image
    shrunk = tensors.grow_mask(mask, -1)
    assert shrunk[0].sum() == 0 and shrunk[1].sum() == 9 and shrunk[1, 2:5, 2:5].all()


def test_erosion_treats_the_outside_of_the_image_as_background(tensors):
    full = torch.ones(1, 5, 5)
    eroded = tensors.grow_mask(full, -1)
    assert eroded.sum() == 9 and eroded[0, 0].sum() == 0 and eroded[0, :, -1].sum() == 0
    assert tensors.grow_mask(full, 1).sum() == 25  # dilation stays inside the image


def test_stack_views_makes_a_batch_or_a_sheet(tensors):
    arrays = [np.full((2, 3, 3), i, dtype=np.float32) for i in range(4)]
    batch = tensors.stack_views(arrays, "batch")
    assert batch.shape == (4, 2, 3, 3) and (batch[2] == 2).all()
    sheet = tensors.stack_views(arrays, "sheet")
    assert sheet.shape == (1, 2, 12, 3)
    assert all((sheet[0, :, 3 * i:3 * (i + 1)] == i).all() for i in range(4))
