import torch
from torch import nn
import numpy as np

from typing import Callable, List

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class BaseRnn(nn.Module):
    """Shared utilities for all three PyTorch RNN variants."""

    def _last_three_months(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x[:, -3:, :]

    def _time_distributed(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:  # noqa: D401
        b, t, h = x.size()
        x = layer(x.reshape(b * t, h)).reshape(b, t, -1)
        return x

    def compare_predictions(
        self,
        x_md: torch.Tensor,  # (B, Q*3, n_monthly)
        x_qd: torch.Tensor,  # (B, Q, n_quarterly)
        x_shift: torch.Tensor,  # (B, Q*3, 1)
        y_true: torch.Tensor,  # (B, Q*3, 1)
        metrics: List[Callable[[np.ndarray, np.ndarray], float]] = [root_mean_squared_error, mean_absolute_error],
    ):
        self.eval()
        with torch.no_grad():
            y_pred = self(x_md, x_qd, x_shift)
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy().repeat(y_pred.shape[-1], -1)

            return [metric(y_true, y_pred) for metric in metrics], y_pred

