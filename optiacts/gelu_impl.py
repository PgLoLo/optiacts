import torch as t

from optiacts.bit_compress import from_bool_tensor, to_bool_tensor


MIN_POINT = -0.751791524693564457457904947


LEFT_Y_MIN = -0.16997120747990369
LEFT_Y_DELTA = 0.16997120747990369
LEFT_ALPHA = 0.44236329315571304
LEFT_BETA = 0.8312705290392689
LEFT_A = -0.2964121012745849
LEFT_B = 0.0012176597919768537


@t.compile
def left_part(y: t.Tensor) -> t.Tensor:
    y = t.clamp((y - LEFT_Y_MIN) / LEFT_Y_DELTA, 0, 1)
    return LEFT_B + LEFT_A * y**LEFT_ALPHA * (1 - y)**LEFT_BETA


RIGHT_Y_MIN = -0.16997120747990369
RIGHT_EXP_MEAN = -2.1198850898439496
RIGHT_EXP_STD = 0.03214673676937606
RIGHT_BIAS = -1.383717971214795
RIGHT_SQR_COEFF = 1.5584201843500274
RIGHT_LINEAR_COEFF = 0.04404574801811055


@t.compile
def right_part(y: t.Tensor) -> t.Tensor:
    y = t.clamp(y - RIGHT_Y_MIN, 0)
    exp = t.exp((RIGHT_EXP_MEAN - y)**3 * RIGHT_EXP_STD)
    poly = RIGHT_BIAS + RIGHT_SQR_COEFF * t.sqrt(y) + RIGHT_LINEAR_COEFF * y
    return 1 + poly * exp


class GELUFn(t.autograd.Function):
    @staticmethod
    def forward(ctx, x: t.Tensor) -> t.Tensor:
        is_left_part = from_bool_tensor(x < MIN_POINT).detach()
        y = t.nn.functional.gelu(x)
        ctx.save_for_backward(is_left_part, y)
        return y

    @staticmethod
    def backward(ctx, output_grad: t.Tensor) -> t.Tensor:
        is_left_part, y = ctx.saved_tensors
        is_left_part = to_bool_tensor(is_left_part, y.shape)
        return output_grad * t.where(is_left_part, left_part(y), right_part(y))


def gelu(x: t.Tensor) -> t.Tensor:
    return GELUFn.apply(x)


class GELU(t.nn.Module):
    def forward(self, x):
        return gelu(x)
