# Huber Loss: smooth interpolation between MAE and MSE.
# Acts like MSE for small errors (|e| < delta) and MAE for large errors.
# This balances MSE metric optimization with MAE's robustness to outliers.
import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(Loss, self).__init__()
        self.delta = 1.0  # Huber threshold

    def forward(
        self,
        pred: Tensor,
        true: Tensor,
        mask: Tensor | None = None,
        **kwargs
    ) -> dict[str, Tensor]:
        # BEGIN adaptor
        if mask is None:
            mask = torch.ones_like(true, device=true.device)
        # END adaptor

        residual = (pred - true) * mask
        abs_res = torch.abs(residual)
        # Huber: quadratic for |e| < delta, linear for |e| >= delta
        huber = torch.where(
            abs_res <= self.delta,
            0.5 * residual ** 2,
            self.delta * (abs_res - 0.5 * self.delta)
        )
        num_eval = mask.sum()
        loss = huber.sum() / (num_eval if num_eval > 0 else 1)

        return {
            "loss": loss,
            "loss_reduction_none": abs_res  # same shape as pred
        }
