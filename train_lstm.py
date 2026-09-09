"""
Training script for the from-scratch LSTM baseline (see lstm_baseline.py),
for comparison against the Time-LLM reprogramming model on the same task.

Example (Step 3 real-world benchmark, matching seq_len/pred_len):
    python train_lstm.py --csv data/electric_motor_temp_profile70.csv \
        --target pm --seq_len 240 --pred_len 60 --epochs 30
"""

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lstm_baseline import LSTMForecaster
from data_provider import TimeSeriesDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    total_mse, total_mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_mse += nn.functional.mse_loss(pred, y, reduction="sum").item()
            total_mae += nn.functional.l1_loss(pred, y, reduction="sum").item()
            n += y.numel()
    return total_mse / n, total_mae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--target", type=str, required=True, help="Target column name")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_path", type=str, default="checkpoints/lstm_baseline.pt")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for init + data shuffling")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | seed: {args.seed}")

    train_ds, val_ds, test_ds = TimeSeriesDataset.train_val_test_split(
        args.csv, args.target, seq_len=args.seq_len, pred_len=args.pred_len
    )
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = LSTMForecaster(
        pred_len=args.pred_len, hidden_size=args.hidden_size, num_layers=args.num_layers,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable params: {n_trainable:,} (all params -- trained from scratch, no frozen backbone)")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.MSELoss()

    resume_path = args.save_path + ".resume"
    start_epoch = 1
    best_val_mse = float("inf")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_mse = ckpt["best_val_mse"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} (best_val_mse so far: {best_val_mse:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)
        val_mse, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mse)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | train_mse={train_loss:.4f} | "
              f"val_mse={val_mse:.4f} | val_mae={val_mae:.4f} | "
              f"lr={lr:.2e} | time={time.time() - t0:.1f}s")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> saved new best checkpoint to {args.save_path}")

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_val_mse": best_val_mse,
        }, resume_path)

    model.load_state_dict(torch.load(args.save_path))
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"\nFINAL TEST RESULTS: MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    if os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    main()
