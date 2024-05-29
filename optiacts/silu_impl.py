import torch as t

from optiacts.bit_compress import from_bool_tensor, to_bool_tensor


MIN_POINT = -1.278464542761073795109358739022980155439


LEFT_Y_MIN = -0.2784645427610738
LEFT_SQRT = -0.5076843705082636
LEFT_POLY = [0.3574942048593756, 0.07963139766917564, 0.21717700759576886]


def left_part(y: t.Tensor) -> t.Tensor:
    y = y - LEFT_Y_MIN
    return LEFT_SQRT * t.sqrt(y) + (LEFT_POLY[0] * y + LEFT_POLY[1]) * y + LEFT_POLY[2]


RIGHT_Y_MIN = -0.2784645427610738
RIGHT_EXP_MEAN = -5.770613302664509
RIGHT_EXP_STD = 0.0026961639850448835
RIGHT_BIAS = -1.3108564021309803
RIGHT_SQR_COEFF = 0.8485896470316523
RIGHT_LINEAR_COEFF = -0.16299051259510922


def right_part(y: t.Tensor) -> t.Tensor:
    y = y - RIGHT_Y_MIN
    exp = t.exp((RIGHT_EXP_MEAN - y)**3 * RIGHT_EXP_STD)
    poly = RIGHT_BIAS + RIGHT_SQR_COEFF * t.sqrt(y) + RIGHT_LINEAR_COEFF * y
    return 1 + poly * exp


@t.compile
def forward(x: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    is_left_part = from_bool_tensor(x < MIN_POINT)
    y = t.nn.functional.silu(x)
    return is_left_part, y


@t.compile
def backward(is_left_part: t.Tensor, y: t.Tensor, output_grad: t.Tensor) -> t.Tensor:
    uncompressed = to_bool_tensor(is_left_part, y.shape)
    deriv = t.where(uncompressed, left_part(y), right_part(y)) * (1 - y) + y
    return output_grad * deriv


class SiLUFn(t.autograd.Function):
    @staticmethod
    def forward(ctx, x: t.Tensor) -> t.Tensor:
        is_left_part, y = forward(x)
        ctx.save_for_backward(is_left_part, y)
        return y

    @staticmethod
    def backward(ctx, output_grad: t.Tensor) -> t.Tensor:
        is_left_part, y = ctx.saved_tensors
        return backward(is_left_part, y, output_grad)


def silu(x: t.Tensor) -> t.Tensor:
    return SiLUFn.apply(x)


class SiLU(t.nn.Module):
    def forward(self, x):
        return silu(x)
