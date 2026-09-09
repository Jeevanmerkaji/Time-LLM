"""
DLinear baseline (Zeng et al., 2022, "Are Transformers Effective for Time
Series Forecasting?") -- included as a second from-scratch reference point
alongside the LSTM baseline. Decomposes the input into a moving-average
trend and a seasonal (residual) component, fits each with a single linear
layer, and sums the two forecasts. Despite its simplicity it is a
well-known strong baseline in this literature, making it a natural check
on whether either from-scratch method's advantage over Time-LLM is
specific to recurrent architectures or holds even more starkly against
something simpler still.

Uses the same RevIN normalization and TimeSeriesDataset windowing as
model.py and lstm_baseline.py so the comparison isolates the forecasting
method itself.
"""

import torch
import torch.nn as nn

from model import RevIN


class MovingAvg(nn.Module):
    """Moving average via 1D avg-pooling, padded so output length == input length."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [batch, seq_len]
        pad_front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        pad_end = x[:, -1:].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2)
        padded = torch.cat([pad_front, x, pad_end], dim=1)
        return self.avg(padded.unsqueeze(1)).squeeze(1)  # [batch, seq_len]


class DLinear(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24, kernel_size: int = 25):
        super().__init__()
        self.revin = RevIN()
        self.decompose = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [batch, seq_len]  -- raw (unnormalized) univariate series
        x = self.revin(x, "norm")
        trend = self.decompose(x)
        seasonal = x - trend
        forecast = self.linear_trend(trend) + self.linear_seasonal(seasonal)
        return self.revin(forecast, "denorm")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
