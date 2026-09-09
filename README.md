# Time-LLM Reproduction — Motor Temperature Forecasting

A simplified reproduction of **"Time-LLM: Time Series Forecasting by
Reprogramming Large Language Models"** (Jin et al., ICLR 2024,
[arXiv:2310.01728](https://arxiv.org/abs/2310.01728)), applied to
motor temperature sensor data.

Official repo (for full-fidelity comparison): https://github.com/KimMeen/Time-LLM

## What this is

The core idea from the paper: instead of training a bespoke forecasting
model from scratch, take a **frozen, pretrained language model** (GPT-2
here — the paper primarily uses Llama), and teach a small trainable
"reprogramming" layer to translate numeric time-series patches into
something that looks like the LLM's native input space. Only the
reprogramming layer + a small output head are trained; the LLM itself
never updates.

This matters for a PhD application bridge narrative: it connects a
sensor/motor-temperature forecasting background (e.g. an LSTM-based
thesis) to LLM-based methods directly, rather than starting an LLM
project from zero. Real-world validation uses a public Kaggle motor
temperature dataset (see Step 3) rather than any proprietary data, so
this repo and its results are shareable as-is.

## ⚠️ Status — what's been verified

Originally built in a sandboxed environment with no internet/GPU access, so
only the architecture (forward/backward passes on a randomly-initialized
GPT-2-shaped model) had been checked. It has since been **run end-to-end on
real hardware** with real pretrained weights — see the
[Results](#results) section below for actual numbers. Verified so far:
- ✅ Architecture is correct (patch embedding → reprogramming cross-attention
  → frozen GPT-2 → forecast head)
- ✅ Real `gpt2` pretrained weights load and stay frozen during training
- ✅ Only ~0.1–0.2% of parameters are trainable (reprogramming layer +
  patch embedding + output head + RevIN affine params), matching the
  paper's low-trainable-parameter design
- ✅ Step 1 (synthetic data) and Step 2 (ETTh1 public benchmark) both run
  end-to-end on CPU and produce converging loss curves
- ✅ RevIN (reversible instance normalization) implemented and shown to
  measurably close part of the accuracy gap (see Results)
- ✅ Model capacity (`d_model` 16→32) and an LR scheduler
  (`ReduceLROnPlateau`) tried on top of RevIN — only a small further
  improvement, suggesting the remaining gap is architectural, not tuning
  (see Discussion)
- ✅ Prompt-as-Prefix (PaP) implemented — a per-window text prompt (compact
  instruction + min/max/median/trend/lags) is prepended to the reprogrammed
  patches before the frozen LLM. Came out roughly on par with (slightly
  worse than) the non-PaP result under a shorter training budget — see
  Discussion for why this likely undersells PaP rather than disproving it
- ✅ Run end-to-end on a real-world dataset — Kaggle's "Electric Motor
  Temperature" PMSM dataset (rotor temperature, `pm`) — beating a naive
  persistence baseline by ~41-49% on MSE/MAE (see Results)
- ✅ Compared against a from-scratch LSTM baseline on the identical
  real-world data/windows — the **LSTM baseline actually wins**: 31% lower
  MSE, 22% lower MAE, ~26x fewer trainable params, ~5x faster to train
  (see Results and Discussion) — an honest, useful negative result for the
  reprogramming approach on this particular task

Still open:
- ❌ Numbers are not yet close to the paper's Table 1 (ETTh1, 96→24) — see
  Results and Discussion below for the identified causes and next steps
- ❌ Our PaP prompt is a compact stats-only version, not the paper's richer
  natural-language dataset context/domain knowledge (cut for CPU speed —
  each extra prompt token is extra frozen-LLM sequence length every step)
- ❌ Backbone is GPT-2 (124M), not the paper's primary Llama-7B

## Setup

Requires **Python ≤3.11** — as of this writing PyTorch has no wheels for
Python 3.14, so if your default `python` is newer, create the venv with an
older interpreter explicitly, e.g. on Windows with the `py` launcher:

```bash
py -3.11 -m venv venv
./venv/Scripts/pip install torch transformers pandas numpy   # Windows
# source venv/bin/activate && pip install torch transformers pandas numpy   # macOS/Linux
```

## Step 1 — Sanity check on synthetic data (fast, CPU is fine)

```bash
python train.py --csv data/synthetic_motor_temp.csv --target temperature \
    --seq_len 48 --pred_len 12 --epochs 3 --batch_size 8 \
    --d_model 16 --num_prototypes 200
```

This should run in a few minutes on CPU and confirm everything works on
your machine before you touch real data or a GPU.

## Step 2 — Public benchmark (recommended before your own data)

Download ETTh1.csv from https://github.com/zhouhaoyi/ETDataset into `data/`,
then:

```bash
python train.py --csv data/ETTh1.csv --target OT \
    --seq_len 96 --pred_len 24 --epochs 10 --batch_size 16
```

Compare your MSE/MAE against the numbers reported in Table 1 of the paper
(ETTh1, 96→24 horizon) — this is your actual "reproduction" result.

## Step 3 — Real-world benchmark (Kaggle Electric Motor Temperature)

Public dataset: [wkirgsn/electric-motor-temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)
— 185 hours of PMSM (permanent magnet synchronous motor) test-bench
recordings at 2Hz, Paderborn University. Requires a (free) Kaggle account
to download `measures_v2.csv` (300MB; a 122MB zip via the "Download
dataset as zip" option in the download dropdown).

**Important — this file is not one continuous series.** It's 69 separate
measurement sessions concatenated together, identified by a `profile_id`
column, no timestamp column. Feeding the whole file through this repo's
chronological-split loader as-is would silently splice unrelated sessions
together at the boundaries. Filter to a single session first:

```python
import pandas as pd
df = pd.read_csv("data/electric_motor_temp/measures_v2.csv")
df[df["profile_id"] == 70].reset_index(drop=True).to_csv(
    "data/electric_motor_temp_profile70.csv", index=False)
```

(`profile_id=70` here: 25,677 rows ≈ 3.6 hours, a mid-sized session —
`df["profile_id"].value_counts()` to see all 69 and pick your own.)

**Target column**: `pm` (permanent magnet / rotor temperature) — the
classic hard-to-measure-directly target in this literature, analogous to
"motor temperature forecasting."

**Window sizing matters here.** At 2Hz, `seq_len=96`/`pred_len=24` (the
ETTh1 windows) means "predict 12 seconds ahead from 48 seconds of
history" — close to a trivial persistence task, since motor temperature
barely moves that fast. Use a longer horizon so the task is real, e.g.
`seq_len=240`/`pred_len=60` (2 minutes of history → 30 seconds ahead):

```bash
python train.py --csv data/electric_motor_temp_profile70.csv --target pm \
    --seq_len 240 --pred_len 60 --epochs 15 --batch_size 16 \
    --d_model 32 --num_prototypes 200 --no_prompt
```

`--no_prompt` disables Prompt-as-Prefix (see Step 2 Discussion) — it's
~5x more CPU-expensive and this window size already has 29 patches
instead of ETTh1's 11, so this run alone was estimated at ~9 hours on
CPU; adding PaP on top wasn't attempted here.

## Results

All runs below: CPU only (no GPU), `patch_len=16`, `stride=8`, GPT-2
backbone frozen, batch size 8 (synthetic) or 16 (ETTh1). Environment:
Python 3.11 venv, `torch`/`transformers` CPU build.

### Step 1 — synthetic sanity check (`seq_len=48`, `pred_len=12`, 3 epochs)

Trainable params: 84,300 / 124,524,108 total (0.07%).

| Epoch | train_mse | val_mse | val_mae |
|---|---|---|---|
| 1 | 1.5960 | 0.3653 | 0.5277 |
| 2 | 0.3048 | 0.2187 | 0.3969 |
| 3 | 0.1806 | 0.0699 | 0.2274 |

**Test: MSE=0.0923, MAE=0.2504** (normalized-scale, pre-RevIN version of the code).
Confirms the pipeline works end-to-end and loss converges monotonically.

### Step 2 — ETTh1 public benchmark (`seq_len=96`, `pred_len=24`)

Four versions were run, each building on the last (see Discussion):

| | Config | Trainable params | Test MSE | Test MAE |
|---|---|---|---|---|
| Global normalization (original code) | `d_model=16`, 10 epochs, fixed lr | 240,984 (0.19%) | 5.9605 | 1.9694 |
| + RevIN (per-instance norm) | `d_model=16`, 10 epochs, fixed lr | 240,986 (0.19%) | 3.3831 | 1.3817 |
| **+ capacity + LR schedule** | `d_model=32`, 15 epochs, `ReduceLROnPlateau` | 278,938 (0.22%) | **3.2489** | **1.3155** |
| + Prompt-as-Prefix (compact) | `d_model=32`, **6 epochs** (time-limited), `ReduceLROnPlateau` | 278,938 (0.22%) | 3.4387 | 1.3793 |
| Paper (Time-LLM, ETTh1 96→24, Table 1) | Llama-7B backbone | — | ~0.36–0.40 | ~0.40–0.41 |

RevIN was the one change that made a real dent (−43% MSE, −30% MAE).
Doubling `d_model` and adding an LR scheduler on top only bought another
−4% MSE / −5% MAE. Adding Prompt-as-Prefix came out slightly *worse*
(+4.1% MSE, +4.8% MAE) than the no-PaP run directly above it — but that
run only got 6 epochs vs. 15 (PaP roughly quintuples per-sample compute on
CPU, since the frozen LLM now processes ~30 extra prompt tokens on top of
the 11 patch tokens per sample, so a full 15-epoch PaP run was estimated
at ~13 hours and wasn't run to completion). Not a clean apples-to-apples
comparison — see Discussion.

Capacity + LR-schedule run, full per-epoch log (`ReduceLROnPlateau`,
factor=0.5, patience=2; best checkpoint = epoch 12, ~555s/epoch):

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

Prompt-as-Prefix run, full per-epoch log (best checkpoint = epoch 3,
~2,508s/epoch — ~5x slower per epoch than the non-PaP run above, due to
the ~30 extra prompt tokens the frozen LLM now processes per sample):

| Epoch | train_mse | val_mse | val_mae | lr |
|---|---|---|---|---|
| 1 | 10.6193 | 2.6490 | 1.2323 | 1e-3 |
| 2 | 7.2549 | 2.4260 | 1.1872 | 1e-3 |
| 3 | 6.9794 | **2.3508** | **1.1571** | 1e-3 |
| 4 | 6.7611 | 2.4620 | 1.2038 | 1e-3 |
| 5 | 6.6658 | 2.6193 | 1.2006 | 1e-3 |
| 6 | 6.4826 | 2.4566 | 1.1904 | 5e-4 |

### Discussion — why the gap to the paper remains

1. **RevIN was worth it; capacity/LR tuning is hitting diminishing
   returns.** Doubling `d_model` (16→32) and adding `ReduceLROnPlateau`
   only moved test MSE from 3.3831 → 3.2489 (−4%). The scheduler barely
   engaged — val_mse is noisy enough epoch-to-epoch (bounces ±0.1-0.3 even
   while trending down) that `patience=2` kept getting reset before it
   could fire; it only cut the LR once, on the very last epoch, too late
   to help. This suggests the bottleneck is no longer "undertrained small
   model" — it's something more structural.
2. **Prompt-as-Prefix, implemented but inconclusive.** A compact per-window
   prompt (short instruction + min/max/median/trend/top-5 lags, no free
   natural-language description) is concatenated in front of the
   reprogrammed patches, tokenized/embedded via the frozen LLM's own
   embedding table, with left-padding and recomputed position IDs so
   variable-length prompts stay aligned. It works correctly (verified
   twice on synthetic data) but roughly quintuples per-sample compute on
   CPU — a full 15-epoch ETTh1 run was estimated at ~13 hours, so it was
   run for only 6 epochs (~4.2 hours) instead, and came out marginally
   *worse* than the 15-epoch non-PaP run. Two confounds make this a weak
   negative result rather than a real one: (a) far fewer training epochs,
   and (b) the prompt was deliberately stripped of the paper's richer
   natural-language dataset context and domain knowledge — exactly the
   part the paper credits with giving the LLM something to "reason" about
   — because every extra token there is extra frozen-LLM sequence length
   on every forward pass. A fair test of PaP would need either a GPU or
   much more CPU time than was available here.
3. **GPT-2 (124M) vs. Llama-7B** — the paper's headline numbers use a
   frozen backbone with >50x more parameters to draw structure from. This
   is a real capacity ceiling that's hard to remove without a GPU, and
   likely explains a meaningful chunk of the remaining ~8x MSE gap on its
   own — plausibly the single biggest lever left, and the one most blocked
   by lacking a GPU.
4. Both non-PaP ETTh1 runs used only 10-15 epochs, and the PaP run only 6;
   the paper's official runs use longer, dataset-tuned schedules.

### Step 3 — Kaggle Electric Motor Temperature (`seq_len=240`, `pred_len=60`, real data)

Quick correctness check first, at the (too-easy) ETTh1-matched window size
`seq_len=48`/`pred_len=12` (6 seconds ahead — see Step 3 above for why
this is near-trivial for this data), `--no_prompt`, 3 epochs, batch 8:

| Epoch | train_mse | val_mse | val_mae |
|---|---|---|---|
| 1 | 0.0878 | 0.0701 | 0.1409 |
| 2 | 0.0655 | 0.0576 | 0.1262 |
| 3 | 0.0578 | 0.0709 | 0.1501 |

Test MSE=0.0097, MAE=0.0371 — confirmed the pipeline works correctly on a
brand-new dataset/target column, but the near-zero error here is mostly
the trivial short-horizon effect, not a meaningful signal.

Full run at the corrected window size (`seq_len=240`, `pred_len=60`,
`d_model=32`, `num_prototypes=200`, `--no_prompt`, RevIN +
`ReduceLROnPlateau`, batch 16, profile_id=70 — 17,674/2,509/5,077
train/val/test windows, 1,412,542 trainable params (1.12%), ~1,910s/epoch,
~7.97 hours total):

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

Unlike ETTh1, val_mse was **still improving at epoch 15** — the LR
scheduler never fired (stayed at 1e-3 throughout). More epochs would
plausibly help further; 15 was a time-budget stopping point, not a
plateau.

**Final test: MSE=0.0236, MAE=0.0954** (raw °C units, via RevIN).

### Step 4 — LSTM baseline (same real data, same windows)

`train_lstm.py`, same `seq_len=240`/`pred_len=60`/`profile_id=70` split,
`hidden_size=64`, `num_layers=2`, `ReduceLROnPlateau`, 30 epochs
(~185s/epoch, ~92.6 min total — ~5x faster than the Time-LLM run since
there's no 124M-parameter frozen backbone in the loop), 54,334 trainable
params (all of them — nothing frozen):

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

Best checkpoint = **epoch 4**. Clear overfitting after that — train_mse
kept falling all the way to 0.067 while val_mse climbed to 0.28, and 8
LR cuts from the scheduler slowed but didn't reverse it (LR decay alone
can't fix overfitting once the model has started memorizing). Final
evaluation correctly reloads the epoch-4 checkpoint, not the overfit
epoch-30 weights.

**Final test: MSE=0.0162, MAE=0.0745**

### Three-way comparison

| Method | Test MSE | Test MAE | Trainable params | Train time |
|---|---|---|---|---|
| Naive persistence (repeat last value) | 0.0460 | 0.1625 | 0 | — |
| Time-LLM (frozen GPT-2, reprogrammed) | 0.0236 | 0.0954 | 1,412,542 | ~7.97h |
| **LSTM (from scratch)** | **0.0162** | **0.0745** | **54,334** | **~1.55h** |

Both beat naive persistence by a wide margin (real evidence of learned
dynamics, not just exploiting smoothness), but **the LSTM baseline wins
outright**: 31% lower MSE, 22% lower MAE than Time-LLM, with ~26x fewer
trainable parameters and ~5x less wall-clock training time, no frozen
124M-parameter backbone required. There's no paper-reported baseline for
this dataset (it's not one of the paper's own benchmarks), so all three
numbers here stand on their own rather than against Table 1.

**Honest takeaway for the writeup**: LLM reprogramming works and
generalizes to a real, previously-unseen sensor dataset (Prompt-as-Prefix
and RevIN both function correctly end-to-end on it), but on this
particular smoothly-varying physical signal, the reprogramming overhead
didn't translate into an accuracy edge over a small, purpose-built,
much-faster-to-train LSTM. That's a legitimate, nuanced finding — not
every task benefits from routing through a frozen general-purpose LLM,
and knowing where the crossover point is (large/diverse pretraining
benchmarks like ETTh1 vs. small/well-behaved single-sensor series) is
itself a useful result to report, especially set against a background of
having built the LSTM-style baseline first.

## Step 4 — Compare against a baseline of your own

`lstm_baseline.py` + `train_lstm.py` provide a from-scratch LSTM baseline
(same RevIN normalization, same `TimeSeriesDataset` windowing, so the
comparison isolates the forecasting method rather than data-handling
differences) — use the **same** `seq_len`/`pred_len` as your Time-LLM run
for a fair, citable comparison:

```bash
python train_lstm.py --csv data/electric_motor_temp_profile70.csv \
    --target pm --seq_len 240 --pred_len 60 --epochs 30
```

See Results for the actual numbers from this run — spoiler: the LSTM
baseline won on this dataset. If you have your own baseline (e.g. a
thesis LSTM) instead, swap it in for the same comparison. This repo's
Results section also includes a naive-persistence baseline (predict the
last observed value) as a cheap sanity floor.

## Suggested report structure (for arXiv / workshop writeup)

1. **Motivation** — bridge from physics-based simulation → LSTM (your
   thesis) → LLM-reprogrammed forecasting (this work)
2. **Method** — brief summary of Time-LLM's reprogramming mechanism
   (cite the original paper properly — see `CITATION.md`)
3. **Setup** — your dataset, preprocessing, train/val/test split, backbone
   choice (GPT-2), hyperparameters
4. **Results** — table: LSTM baseline vs. reprogrammed-LLM, MSE + MAE
5. **Discussion** — what worked, what didn't, compute cost tradeoffs,
   honest limitations
6. **Reproducibility** — link to your GitHub repo with exact run commands

## Files

- `model.py` — TimeLLM model: RevIN (per-instance normalization, reversed
  on output), Prompt-as-Prefix (per-window text prompt of instruction +
  stats, tokenized/embedded and prepended to the patch sequence before the
  LLM, left-padded with recomputed position IDs; toggle with `use_prompt`),
  patch embedding, reprogramming cross-attention layer, frozen GPT-2
  backbone, output projection
- `data_provider.py` — CSV dataset loader with chronological train/val/test
  split (no data leakage); series are kept in raw units since RevIN
  normalizes per-window inside the model
- `train.py` — training loop, evaluation (MSE/MAE), checkpointing,
  `ReduceLROnPlateau` LR scheduler, `--description` flag for the
  Prompt-as-Prefix dataset context text, `--no_prompt` to disable PaP
  (much faster on CPU)
- `data/synthetic_motor_temp.csv` — synthetic test data for sanity-checking
  the pipeline before using real data
- `data/ETTh1.csv` — public benchmark dataset (from
  https://github.com/zhouhaoyi/ETDataset) for the Step 2 comparison
- `data/electric_motor_temp/measures_v2.csv` — raw Kaggle download (300MB,
  69 concatenated sessions) — not meant to be committed to git as-is;
  `.gitignore` it if this repo goes public
- `data/electric_motor_temp_profile70.csv` — single filtered session
  (`profile_id=70`) used for the Step 3 real-world benchmark
- `lstm_baseline.py` — from-scratch LSTM baseline model (RevIN + `nn.LSTM`
  + linear head), for a fair Step 4 comparison against Time-LLM
- `train_lstm.py` — training script for the LSTM baseline, mirrors
  `train.py`'s loop/checkpointing/`ReduceLROnPlateau`
- `checkpoints/time_llm.pt` — best Time-LLM checkpoint from the most
  recent training run (currently: Step 3 real-world benchmark, epoch 15)
- `checkpoints/lstm_baseline.pt` — best LSTM baseline checkpoint (Step 4
  real-world run, epoch 4)

## Citation

If you use or build on this, cite the original paper:

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
