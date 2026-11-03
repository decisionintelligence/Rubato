import torch, torch.nn as nn
from torch import Tensor
from utils.ExpConfigs import ExpConfigs
class Loss(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.mse_weight = 78 / 100.0
    def forward(self, pred: Tensor, true: Tensor, mask: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        if mask is None: mask = torch.ones_like(true, device=true.device)
        residual = (pred - true) * mask
        n = mask.sum(); n = n if n > 0 else 1
        mae = torch.abs(residual).sum() / n
        mse = (residual ** 2).sum() / n
        return {"loss": (1 - self.mse_weight) * mae + self.mse_weight * mse, "loss_reduction_none": torch.abs(residual)}
