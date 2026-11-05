"""Turn the library's NumPy results into ComfyUI outputs (torch tensors).

torch is imported inside the functions, so that importing the pack stays cheap.
"""

from __future__ import annotations

import numpy as np


def to_tensor(array: np.ndarray):
    """``float32`` torch tensor of ``array`` (no copy when it already is contiguous ``float32``)."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def grow_mask(mask, pixels: int):
    """Dilate (``pixels > 0``) or erode (``pixels < 0``) a ``(N, H, W)`` mask with a square kernel."""
    if not pixels:
        return mask
    import torch.nn.functional as F

    radius = abs(int(pixels))
    x = mask[:, None]
    if pixels > 0:
        y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    else:
        # max_pool2d pads with -inf; after negation that would count everything
        # outside the image as inside the mask, so pad with zeros explicitly.
        x = F.pad(x, (radius, radius, radius, radius), value=0.0)
        y = -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1)
    return y[:, 0]
