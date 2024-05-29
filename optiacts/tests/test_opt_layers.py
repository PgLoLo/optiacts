import pytest
import torch as t
from torch.nn.functional import gelu, silu

import optiacts


@pytest.mark.parametrize(
    'true_approx_fns', [(gelu, optiacts.GELU()), (silu, optiacts.SiLU()), (gelu, optiacts.gelu), (silu, optiacts.silu)]
)
@pytest.mark.parametrize('limit', [3, 100])
def test_optimized_layer(true_approx_fns, limit: float):
    true_fn, approx_fn = true_approx_fns
    xs_true = t.linspace(-limit, limit, 2 ** 20, requires_grad=True)
    ys_true = true_fn(xs_true)
    ys_true.sum().backward()

    xs_approx = t.linspace(-limit, limit, 2 ** 20, requires_grad=True)
    ys_approx = approx_fn(xs_approx)
    ys_approx.sum().backward()

    assert t.allclose(ys_true, ys_approx, atol=1e-6)
    assert t.allclose(xs_true.grad, xs_approx.grad, atol=2e-2)
