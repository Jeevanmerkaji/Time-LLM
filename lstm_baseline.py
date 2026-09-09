"""
From-scratch LSTM baseline for the Step 3 real-world benchmark, for a fair
comparison against the Time-LLM reprogramming model in model.py. Uses the
same RevIN normalization and the same TimeSeriesDataset windowing, so the
comparison isolates the forecasting method itself (frozen-LLM reprogramming
vs. a trained-from-scratch recurrent model) rather than normalization or
data-splitting differences.
"""

import torch
import torch.nn as nn

from model import RevIN


class LSTMForecaster(nn.Module):
    def __init__(self, pred_len: int = 60, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.revin = RevIN()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: [batch, seq_len]  -- raw (unnormalized) univariate series
        x = self.revin(x, "norm")
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        _, (h_n, _) = self.lstm(x)
        forecast = self.output_proj(h_n[-1])  # [batch, pred_len], still normalized
        return self.revin(forecast, "denorm")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
