import torch
import torch.nn as nn
from torch import Tensor
from utils.ExpConfigs import ExpConfigs

class Loss(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(Loss, self).__init__()
        self.mse_weight = 60 / 100.0  # 60% MSE + 40% MAE

    def forward(self, pred: Tensor, true: Tensor, mask: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        if mask is None:
            mask = torch.ones_like(true, device=true.device)
        residual = (pred - true) * mask
        num_eval = mask.sum()
        num_eval = num_eval if num_eval > 0 else 1
        mae = torch.abs(residual).sum() / num_eval
        mse = (residual ** 2).sum() / num_eval
        loss = (1 - self.mse_weight) * mae + self.mse_weight * mse
        return {"loss": loss, "loss_reduction_none": torch.abs(residual)}
