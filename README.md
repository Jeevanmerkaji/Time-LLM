# Time-LLM Reproduction — Motor Temperature Forecasting

**Does reprogramming a frozen LLM beat a small from-scratch model at real-world sensor forecasting?** A from-scratch reproduction of [Time-LLM](https://arxiv.org/abs/2310.01728) (Jin et al., ICLR 2024), evaluated against LSTM and DLinear baselines across multiple real-world sessions and seeds — not just one lucky run.

[![Paper](https://img.shields.io/badge/paper-PDF-red)](arxiv_paper/TimeLLM.pdf) [![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#quickstart) [![Original Paper](https://img.shields.io/badge/original-Time--LLM-lightgrey)](https://arxiv.org/abs/2310.01728)

> **Key finding:** Across 3 independent real-world sessions × 3 seeds (27 runs total), **no single method wins consistently.** LSTM wins one session, DLinear/Time-LLM tie on another, and Time-LLM wins the third outright — a genuine split verdict, not a clean win for either the reprogrammed-LLM approach or the classic baseline. Full details in [`PAPER_DRAFT.md`](PAPER_DRAFT.md) and the [compiled paper](arxiv_paper/TimeLLM.pdf).

---

## Table of contents

- [Overview](#overview)
- [Headline results — full-rigor multi-session matrix](#full-rigor-results)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Documentation](#documentation)
- [Citation](#citation)
- [Full experimental walkthrough (single-session pilot)](#full-experimental-walkthrough-single-session-pilot)

---

## Overview

Time-LLM's core idea: instead of training a bespoke forecasting model from scratch, take a **frozen, pretrained language model** (GPT-2 here — the original paper primarily uses Llama) and train only a small "reprogramming" layer that translates numeric time-series patches into the LLM's native input space. Only that reprogramming layer plus a small output head are trained; the LLM itself never updates.

This project reproduces that mechanism end-to-end (patch embedding → reprogramming cross-attention → frozen GPT-2 → forecast head) and validates it on a real, previously-unseen dataset: Kaggle's [Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) PMSM dataset. Rather than a single train/test run, the core contribution here is a **full-rigor experiment matrix** — 3 real recording sessions × 3 methods × 3 seeds — built specifically to test whether a single-session comparison (the norm in a lot of applied forecasting writeups) actually generalizes.

## Full-Rigor Results

3 real Kaggle sessions (`profile_id` 70, 62, 44) × 3 methods (DLinear, LSTM, Time-LLM) × 3 seeds = **27 runs, all complete.**

| Session | Winner (test MSE) | Margin |
|---|---|---|
| `profile_id=70` | **LSTM** | 31% lower MSE than Time-LLM |
| `profile_id=62` | DLinear *(near-tie)* | all three methods within noise of a near-zero error floor |
| `profile_id=44` | **Time-LLM** | 27% lower MSE than LSTM, 10% lower than DLinear |

| Cross-session mean | Test MSE | Test MAE | Sessions won |
|---|---|---|---|
| DLinear | 0.0348 | 0.0817 | 1/3 |
| LSTM | 0.0337 | **0.0581** | 1/3 |
| **Time-LLM** | **0.0290** | 0.0793 | 1/3 |

**Each method wins exactly one session.** Time-LLM has the lowest mean MSE, LSTM has the lowest mean MAE — a split verdict, not a clean win for either approach. The methodological takeaway: a single-session comparison (even a carefully-run one) can produce a method-ranking claim that reverses on a second, equally legitimate session of the same dataset.

Compute cost is the other consistent result: Time-LLM training took **4–16× longer** than LSTM (1.5–8.1h vs. 0.5–1.6h) across every session.

Full per-session tables, discussion, caveats on the cross-session aggregation, and limitations: [`PAPER_DRAFT.md`](PAPER_DRAFT.md) · [compiled PDF](arxiv_paper/TimeLLM.pdf) · raw data in [`results_full_rigor.jsonl`](results_full_rigor.jsonl).

## Quickstart

Requires **Python ≤3.11** — PyTorch has no wheels for very recent Python versions, so use an older interpreter explicitly if your default `python` is newer.

```bash
python3.11 -m venv venv
source venv/bin/activate           # venv\Scripts\activate on Windows
pip install torch transformers pandas numpy
```

Sanity-check the pipeline on synthetic data (a few minutes on CPU):

```bash
python train.py --csv data/synthetic_motor_temp.csv --target temperature \
    --seq_len 48 --pred_len 12 --epochs 3 --batch_size 8 \
    --d_model 16 --num_prototypes 200
```

Reproduce the full-rigor matrix (resumable — safe to interrupt and rerun):

```bash
python run_full_rigor.py
python aggregate_results.py   # regenerate the summary tables above
```

For the public ETTh1 benchmark, real-world data preparation, and the single-session baseline comparison, see the [full walkthrough](#full-experimental-walkthrough-single-session-pilot) below.

## Repository structure

| File | Purpose |
|---|---|
| `model.py` | Time-LLM model: RevIN, patch embedding, reprogramming cross-attention, frozen GPT-2 backbone, output projection |
| `data_provider.py` | CSV dataset loader with chronological train/val/test split |
| `train.py` / `train_lstm.py` / `train_dlinear.py` | Training loops for Time-LLM, LSTM baseline, DLinear baseline |
| `lstm_baseline.py` / `dlinear_baseline.py` | Baseline model definitions |
| `run_full_rigor.py` | Orchestrates the 27-run matrix, resumable |
| `aggregate_results.py` | Computes mean ± std per (session, method), writes `results_full_rigor_summary.md` |
| `results_full_rigor.jsonl` | Raw per-run results (one JSON record each) |
| `PAPER_DRAFT.md` | Full multi-session paper writeup |
| `REPORT.md` | Original single-session detailed report |
| `arxiv_paper/` | LaTeX source (`main.tex`, `references.bib`) + compiled `TimeLLM.pdf` |
| `data/` | Synthetic, ETTh1, and pre-filtered Kaggle motor-temperature CSVs |

## Documentation

- **[`README.md`](README.md)** (this file) — quickstart and headline results
- **[`PAPER_DRAFT.md`](PAPER_DRAFT.md)** — full multi-session, multi-seed paper writeup
- **[`REPORT.md`](REPORT.md)** — original single-session pilot report (superseded by the above, kept for history)
- **[`arxiv_paper/`](arxiv_paper/)** — submission-ready LaTeX source and compiled PDF

## Citation

If you use or build on this work, cite the original paper:

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

---

## Full experimental walkthrough (single-session pilot)

<details>
<summary><strong>Click to expand</strong> — step-by-step tutorial (synthetic data → ETTh1 benchmark → real-world data → single-session LSTM comparison), including every per-epoch training log.</summary>

### Status — what's been verified

- ✅ Architecture is correct (patch embedding → reprogramming cross-attention → frozen GPT-2 → forecast head)
- ✅ Real `gpt2` pretrained weights load and stay frozen during training
- ✅ Only ~0.1–1.1% of parameters are trainable, matching the paper's low-trainable-parameter design
- ✅ Synthetic data and the ETTh1 public benchmark both run end-to-end on CPU and produce converging loss curves
- ✅ RevIN (reversible instance normalization) implemented and shown to measurably close part of the accuracy gap
- ✅ Model capacity (`d_model` 16→32) and an LR scheduler (`ReduceLROnPlateau`) tried on top of RevIN — only a small further improvement, suggesting the remaining gap is architectural, not tuning
- ✅ Prompt-as-Prefix (PaP) implemented — a per-window text prompt (compact instruction + min/max/median/trend/lags) prepended to the reprogrammed patches. Came out roughly on par with (slightly worse than) the non-PaP result under a shorter training budget
- ✅ Run end-to-end on real-world Kaggle motor-temperature data — beating naive persistence by ~41–49% on MSE/MAE
- ✅ Compared against a from-scratch LSTM baseline on one real-world session (`profile_id=70`) — in that single-session pilot, the **LSTM baseline won**: 31% lower MSE, 22% lower MAE, ~26x fewer trainable params, ~5x faster to train
- ⚠️ **This single-session result does not hold up uniformly** — see [Full-Rigor Results](#full-rigor-results) above for the full 3-session, 3-seed matrix that revises this takeaway

Still open:
- ❌ Numbers are not yet close to the paper's Table 1 (ETTh1, 96→24) — likely dominated by backbone size (GPT-2 124M vs. the paper's Llama-7B)
- ❌ The PaP prompt here is a compact stats-only version, not the paper's richer natural-language dataset context (cut for CPU speed)
- ❌ Backbone is GPT-2 (124M), not the paper's primary Llama-7B

### Step 1 — Sanity check on synthetic data

```bash
python train.py --csv data/synthetic_motor_temp.csv --target temperature \
    --seq_len 48 --pred_len 12 --epochs 3 --batch_size 8 \
    --d_model 16 --num_prototypes 200
```

Trainable params: 84,300 / 124,524,108 total (0.07%).

| Epoch | train_mse | val_mse | val_mae |
|---|---|---|---|
| 1 | 1.5960 | 0.3653 | 0.5277 |
| 2 | 0.3048 | 0.2187 | 0.3969 |
| 3 | 0.1806 | 0.0699 | 0.2274 |

**Test: MSE=0.0923, MAE=0.2504** (normalized-scale, pre-RevIN version of the code). Confirms the pipeline works end-to-end and loss converges monotonically.

### Step 2 — Public benchmark (ETTh1)

Download `ETTh1.csv` from [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) into `data/`, then:

```bash
python train.py --csv data/ETTh1.csv --target OT \
    --seq_len 96 --pred_len 24 --epochs 10 --batch_size 16
```

Four versions were run, each building on the last:

| | Config | Trainable params | Test MSE | Test MAE |
|---|---|---|---|---|
| Global normalization (original code) | `d_model=16`, 10 epochs, fixed lr | 240,984 (0.19%) | 5.9605 | 1.9694 |
| + RevIN (per-instance norm) | `d_model=16`, 10 epochs, fixed lr | 240,986 (0.19%) | 3.3831 | 1.3817 |
| **+ capacity + LR schedule** | `d_model=32`, 15 epochs, `ReduceLROnPlateau` | 278,938 (0.22%) | **3.2489** | **1.3155** |
| + Prompt-as-Prefix (compact) | `d_model=32`, **6 epochs** (time-limited), `ReduceLROnPlateau` | 278,938 (0.22%) | 3.4387 | 1.3793 |
| Paper (Time-LLM, ETTh1 96→24, Table 1) | Llama-7B backbone | — | ~0.36–0.40 | ~0.40–0.41 |

RevIN was the one change that made a real dent (−43% MSE, −30% MAE). Doubling `d_model` and adding an LR scheduler on top only bought another −4% MSE / −5% MAE. Adding Prompt-as-Prefix came out slightly *worse* (+4.1% MSE, +4.8% MAE) than the no-PaP run directly above it — but that run only got 6 epochs vs. 15 (PaP roughly quintuples per-sample compute on CPU, since the frozen LLM now processes ~30 extra prompt tokens on top of the 11 patch tokens per sample; a full 15-epoch PaP run was estimated at ~13 hours and wasn't run to completion).

<details>
<summary>Full per-epoch logs (capacity+LR-schedule run, and Prompt-as-Prefix run)</summary>

Capacity + LR-schedule run (`ReduceLROnPlateau`, factor=0.5, patience=2; best checkpoint = epoch 12, ~555s/epoch):

| Epoch | train_mse | val_mse | val_mae | lr |
|---|---|---|---|---|
| 1 | 11.4216 | 3.3496 | 1.4265 | 1e-3 |
| 2 | 8.5125 | 3.3938 | 1.4045 | 1e-3 |
| 3 | 7.7575 | 4.4259 | 1.5324 | 1e-3 |
| 4 | 7.2360 | 2.6287 | 1.2264 | 1e-3 |
| 5 | 7.1915 | 2.7420 | 1.2590 | 1e-3 |
| 6 | 6.9770 | 2.7128 | 1.2320 | 1e-3 |
| 7 | 6.8515 | 2.4500 | 1.1584 | 1e-3 |
| 8 | 6.8209 | 2.5056 | 1.1854 | 1e-3 |
| 9 | 6.6753 | 2.5410 | 1.2010 | 1e-3 |
| 10 | 6.6434 | 2.3274 | 1.1523 | 1e-3 |
| 11 | 6.7220 | 2.3865 | 1.1803 | 1e-3 |
| 12 | 6.5648 | **2.3238** | **1.1441** | 1e-3 |
| 13 | 6.5302 | 2.4114 | 1.1668 | 1e-3 |
| 14 | 6.5298 | 2.3395 | 1.1451 | 1e-3 |
| 15 | 6.5649 | 2.4079 | 1.1779 | 5e-4 |

Prompt-as-Prefix run (best checkpoint = epoch 3, ~2,508s/epoch — ~5x slower per epoch than the non-PaP run, due to the ~30 extra prompt tokens processed per sample):

| Epoch | train_mse | val_mse | val_mae | lr |
|---|---|---|---|---|
| 1 | 10.6193 | 2.6490 | 1.2323 | 1e-3 |
| 2 | 7.2549 | 2.4260 | 1.1872 | 1e-3 |
| 3 | 6.9794 | **2.3508** | **1.1571** | 1e-3 |
| 4 | 6.7611 | 2.4620 | 1.2038 | 1e-3 |
| 5 | 6.6658 | 2.6193 | 1.2006 | 1e-3 |
| 6 | 6.4826 | 2.4566 | 1.1904 | 5e-4 |

</details>

**Discussion — why the gap to the paper remains:**

1. **RevIN was worth it; capacity/LR tuning is hitting diminishing returns.** Doubling `d_model` (16→32) and adding `ReduceLROnPlateau` only moved test MSE from 3.3831 → 3.2489 (−4%). The scheduler barely engaged — val_mse is noisy enough epoch-to-epoch that `patience=2` kept getting reset; it only cut the LR once, on the very last epoch, too late to help.
2. **Prompt-as-Prefix, implemented but inconclusive.** The mechanism is verified correct (left-padding, recomputed position IDs, correct prefix/patch masking) but two confounds make the comparison weak: far fewer training epochs, and a deliberately stripped-down prompt lacking the paper's richer natural-language dataset context. A fair test needs either a GPU or substantially more CPU time.
3. **GPT-2 (124M) vs. Llama-7B** — the paper's headline numbers use a backbone with >50x more parameters. This is a real capacity ceiling that's hard to remove without a GPU, and likely explains a meaningful chunk of the remaining ~8x MSE gap.
4. Both non-PaP ETTh1 runs used only 10-15 epochs, and the PaP run only 6; the paper's official runs use longer, dataset-tuned schedules.

### Step 3 — Real-world benchmark (Kaggle Electric Motor Temperature)

Public dataset: [wkirgsn/electric-motor-temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) — 185 hours of PMSM test-bench recordings at 2Hz, Paderborn University. Requires a free Kaggle account to download `measures_v2.csv` (300MB).

**Important — this file is not one continuous series.** It's 69 separate measurement sessions concatenated together, identified by a `profile_id` column, no timestamp column. Filter to a single session first:

```python
import pandas as pd
df = pd.read_csv("data/electric_motor_temp/measures_v2.csv")
df[df["profile_id"] == 70].reset_index(drop=True).to_csv(
    "data/electric_motor_temp_profile70.csv", index=False)
```

**Target column**: `pm` (permanent magnet / rotor temperature). **Window sizing matters**: at 2Hz, the ETTh1-matched `seq_len=96`/`pred_len=24` is close to a trivial persistence task, since motor temperature barely moves that fast. Use a longer horizon, e.g. `seq_len=240`/`pred_len=60` (2 minutes of history → 30 seconds ahead):

```bash
python train.py --csv data/electric_motor_temp_profile70.csv --target pm \
    --seq_len 240 --pred_len 60 --epochs 15 --batch_size 16 \
    --d_model 32 --num_prototypes 200 --no_prompt
```

<details>
<summary>Full per-epoch logs and correctness check</summary>

Quick correctness check first, at the (too-easy) ETTh1-matched window size `seq_len=48`/`pred_len=12`, `--no_prompt`, 3 epochs, batch 8:

| Epoch | train_mse | val_mse | val_mae |
|---|---|---|---|
| 1 | 0.0878 | 0.0701 | 0.1409 |
| 2 | 0.0655 | 0.0576 | 0.1262 |
| 3 | 0.0578 | 0.0709 | 0.1501 |

Test MSE=0.0097, MAE=0.0371 — confirmed the pipeline works correctly, but the near-zero error is mostly the trivial short-horizon effect.

Full run at the corrected window size (`seq_len=240`, `pred_len=60`, `d_model=32`, `num_prototypes=200`, `--no_prompt`, RevIN + `ReduceLROnPlateau`, batch 16, `profile_id=70` — 17,674/2,509/5,077 train/val/test windows, 1,412,542 trainable params (1.12%), ~1,910s/epoch, ~7.97 hours total):

| Epoch | train_mse | val_mse | val_mae | lr |
|---|---|---|---|---|
| 1 | 0.5691 | 0.5617 | 0.5863 | 1e-3 |
| 2 | 0.4135 | 0.8334 | 0.7340 | 1e-3 |
| 3 | 0.3376 | 0.3459 | 0.4409 | 1e-3 |
| 4 | 0.3390 | 0.2086 | 0.3222 | 1e-3 |
| 5 | 0.2339 | 0.1791 | 0.2856 | 1e-3 |
| 6 | 0.2356 | 0.1975 | 0.3170 | 1e-3 |
| 7 | 0.2297 | 0.2457 | 0.3780 | 1e-3 |
| 8 | 0.2018 | 0.1790 | 0.2813 | 1e-3 |
| 9 | 0.2034 | 0.1762 | 0.2712 | 1e-3 |
| 10 | 0.1973 | 0.2003 | 0.3198 | 1e-3 |
| 11 | 0.1869 | 0.1744 | 0.2769 | 1e-3 |
| 12 | 0.1740 | 0.1678 | 0.2802 | 1e-3 |
| 13 | 0.1703 | 0.1659 | 0.2850 | 1e-3 |
| 14 | 0.1671 | 0.1648 | 0.2757 | 1e-3 |
| 15 | 0.1615 | **0.1622** | **0.2691** | 1e-3 |

Unlike ETTh1, val_mse was **still improving at epoch 15** — the LR scheduler never fired. More epochs would plausibly help further; 15 was a time-budget stopping point, not a plateau.

**Final test: MSE=0.0236, MAE=0.0954** (raw °C units, via RevIN).

</details>

### Step 4 — LSTM baseline (single-session pilot)

```bash
python train_lstm.py --csv data/electric_motor_temp_profile70.csv \
    --target pm --seq_len 240 --pred_len 60 --epochs 30
```

`hidden_size=64`, `num_layers=2`, `ReduceLROnPlateau`, 30 epochs (~185s/epoch, ~92.6 min total), 54,334 trainable params (all of them — nothing frozen).

<details>
<summary>Full per-epoch log</summary>

| Epoch | train_mse | val_mse | val_mae | lr |
|---|---|---|---|---|
| 1 | 0.1901 | 0.1475 | 0.2455 | 1e-3 |
| 2 | 0.1472 | 0.1426 | 0.2361 | 1e-3 |
| 3 | 0.1397 | 0.1418 | 0.2363 | 1e-3 |
| 4 | 0.1310 | **0.1331** | **0.2264** | 1e-3 |
| 5 | 0.1228 | 0.1416 | 0.2372 | 1e-3 |
| 6 | 0.1162 | 0.1470 | 0.2304 | 1e-3 |
| 7 | 0.1123 | 0.1463 | 0.2510 | 5e-4 |
| 8 | 0.1037 | 0.1555 | 0.2424 | 5e-4 |
| 9 | 0.1001 | 0.1445 | 0.2279 | 5e-4 |
| 10 | 0.0990 | 0.1526 | 0.2417 | 2.5e-4 |
| 11–30 | 0.0912 → 0.0673 | 0.1822 → 0.2826 | 0.2527 → 0.3119 | 2.5e-4 → 3.91e-6 |

Best checkpoint = **epoch 4**. Clear overfitting after that — train_mse kept falling to 0.067 while val_mse climbed to 0.28, and 8 LR cuts slowed but didn't reverse it. Final evaluation correctly reloads the epoch-4 checkpoint.

**Final test: MSE=0.0162, MAE=0.0745**

</details>

### Single-session three-way comparison

| Method | Test MSE | Test MAE | Trainable params | Train time |
|---|---|---|---|---|
| Naive persistence (repeat last value) | 0.0460 | 0.1625 | 0 | — |
| Time-LLM (frozen GPT-2, reprogrammed) | 0.0236 | 0.0954 | 1,412,542 | ~7.97h |
| **LSTM (from scratch)** | **0.0162** | **0.0745** | **54,334** | **~1.55h** |

Both beat naive persistence by a wide margin, but on this single session, the LSTM baseline wins outright: 31% lower MSE, 22% lower MAE, ~26x fewer trainable parameters, ~5x less training time. **This is exactly the result that the full-rigor 3-session matrix (see top of README) shows does not hold up uniformly** — it replicates on this session but reverses on another.

To run your own baseline comparison against Time-LLM (same RevIN normalization, same windowing, so the comparison isolates the forecasting method):

```bash
python train_lstm.py --csv data/electric_motor_temp_profile70.csv \
    --target pm --seq_len 240 --pred_len 60 --epochs 30
```

</details>
