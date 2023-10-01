import numbers
from functools import partial
from typing import Union, List

import torch
from torch import Tensor, Size
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LayerNorm(nn.Module):
    # NOTE: taken from official pytorch implementation and modified
    # to allow revoval of gain and bias independently

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 0.00001,
        elementwise_gain: bool = True,
        elementwise_bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_gain = elementwise_gain
        self.elementwise_bias = elementwise_bias

        if self.elementwise_gain:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)

        if self.elementwise_bias:
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_gain:
            with torch.no_grad():
                self.weight.fill_(1.0)

        if self.elementwise_bias:
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_gain={elementwise_gain}, "
            "elementwise_bias={elementwise_bias}".format(**self.__dict__)
        )


class RmsNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps} ".format(**self.__dict__)


def get_norm_class(model_norm):
    if model_norm == "default_layer_norm":
        return torch.nn.LayerNorm

    elif model_norm == "gain_only_layer_norm":
        return partial(LayerNorm, elementwise_gain=True, elementwise_bias=False)

    elif model_norm == "no_wb_layer_norm":
        return partial(LayerNorm, elementwise_gain=False, elementwise_bias=False)

    elif model_norm == "rms_norm":
        return RmsNorm

    else:
        raise ValueError(f"Unsupported model-norm: {model_norm}")
