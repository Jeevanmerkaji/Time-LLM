"""
Training script for the Time-LLM reproduction.

Example (public benchmark):
    python train.py --csv data/ETTh1.csv --target OT --epochs 10

Example (your own motor temperature data):
    python train.py --csv data/motor_temp.csv --target temperature --epochs 20 \
        --seq_len 96 --pred_len 24

Only the reprogramming layer + patch embedding + output head are trained.
The GPT-2 backbone stays frozen, matching the paper's design.
"""

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import TimeLLM
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
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--num_prototypes", type=int, default=1000)
    ap.add_argument("--llm_name", type=str, default="gpt2")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_path", type=str, default="checkpoints/time_llm.pt")
    ap.add_argument("--description", type=str,
                     default="Time series forecasting.",
                     help="Short dataset context used in the Prompt-as-Prefix text "
                          "(keep brief -- every token adds frozen-LLM sequence length)")
    ap.add_argument("--no_prompt", action="store_true",
                     help="Disable Prompt-as-Prefix (much faster on CPU; patches only, no text prompt)")
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

    model = TimeLLM(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        num_prototypes=args.num_prototypes,
        llm_name=args.llm_name,
        description=args.description,
        use_prompt=not args.no_prompt,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / Total params: {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}% trainable)")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.MSELoss()

    # Resume support: this repro has repeatedly lost hours of an ~8h run to
    # unexpected interruptions (power settings, background-task kills), so
    # every epoch's full state is checkpointed -- not just the best-val
    # weights -- letting a restarted run pick up mid-training instead of
    # starting over.
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
    print("Compare these numbers against your LSTM thesis baseline "
          "(same seq_len/pred_len setup) for your writeup.")

    if os.path.exists(resume_path):
        os.remove(resume_path)  # run completed normally -- don't let a future run "resume" into it


if __name__ == "__main__":
    main()
