import numpy as np
import torch as t


@t.compile
def to_bool_tensor(data, shape) -> t.Tensor:
    result = ((data[:, None] >> t.arange(8, device=data.device, dtype=t.uint8)) & 1).bool().view(-1)
    numel = np.prod(shape)
    if result.numel() != numel:
        result = result[:numel]
    return result.reshape(shape)


@t.compile
def from_bool_tensor(other: t.Tensor) -> t.Tensor:
    other = other.view(-1)
    if other.numel() % 8 != 0:
        other = t.constant_pad_nd(other, (0, 8 - other.numel() % 8), 0)
    size = len(other) // 8
    data = (
        other.reshape(size, 8).byte() << t.arange(8, device=other.device, dtype=t.uint8)
    ).sum(axis=1, dtype=t.uint8)
    return data
