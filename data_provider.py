"""
Dataset loader for Time-LLM reproduction.

Works two ways:
1. Public benchmark: point to an ETT-style CSV (timestamp + numeric columns)
   Download from: https://github.com/zhouhaoyi/ETDataset (ETTh1.csv is a good start)
2. Your own data: any CSV with a datetime column + one target numeric column
   (e.g. your Schaeffler motor temperature series, or acoustic emission signal).
   Just point `target_col` at the right column name.

Series are kept in raw (unnormalized) units here — the model applies RevIN
(reversible instance normalization) internally per input window and reverses
it on the output, so there's no dataset-level normalization to fit or leak.

Usage:
    ds = TimeSeriesDataset("data/ETTh1.csv", target_col="OT",
                            seq_len=96, pred_len=24)
    train_ds, val_ds, test_ds = ds.train_val_test_split(...)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, target_col, seq_len=96, pred_len=24):
        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(
                f"Column '{target_col}' not found. Available columns: {list(df.columns)}"
            )
        self.series = df[target_col].astype(float).values
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    @staticmethod
    def train_val_test_split(csv_path, target_col, seq_len=96, pred_len=24,
                              train_frac=0.7, val_frac=0.1):
        """Splits chronologically (no shuffling across time!)."""
        df = pd.read_csv(csv_path)
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        def make_split(start, end):
            sub_df = df.iloc[max(0, start - seq_len): end].reset_index(drop=True)
            tmp_ds = TimeSeriesDataset.__new__(TimeSeriesDataset)
            tmp_ds.series = sub_df[target_col].astype(float).values
            tmp_ds.seq_len, tmp_ds.pred_len = seq_len, pred_len
            return tmp_ds

        train_ds = make_split(0, train_end)
        val_ds = make_split(train_end, val_end)
        test_ds = make_split(val_end, n)
        return train_ds, val_ds, test_ds
