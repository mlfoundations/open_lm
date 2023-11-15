import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss


class CrossEntropyLossWithZLoss(CrossEntropyLoss):
    def __init__(
        self,
        eps: float = 1e-4,
        weight: Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target) + self.eps * torch.square(torch.logsumexp(input, dim=-1)).mean()
