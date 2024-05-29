import pytest
import torch as t

from optiacts.bit_compress import from_bool_tensor, to_bool_tensor


@pytest.mark.parametrize(
    'shape', [1, 2, 3, 128, 129, 126, 127, 256, 1024, (8, 8), (2, 16), (16, 2), (8, 2, 8), (16, 2, 4), (8, 127)]
)
def test_bit_compression(shape):
    original = t.randn(shape) > 0
    compressed = from_bool_tensor(original)
    assert t.all(to_bool_tensor(compressed, shape) == original)
