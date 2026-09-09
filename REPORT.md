# Reprogramming a Frozen LLM for Motor Temperature Forecasting: A Reproduction and Honest Baseline Comparison

*Draft report — Time-LLM reproduction project*

## Abstract

We reproduce a simplified version of Time-LLM (Jin et al., ICLR 2024), a
method that forecasts time series by reprogramming a frozen, pretrained
language model rather than training a forecaster from scratch. Using a
GPT-2 (124M) backbone on CPU-only hardware, we implement the core
reprogramming mechanism, reversible instance normalization (RevIN), and
Prompt-as-Prefix (PaP) text conditioning, and evaluate on both the paper's
own ETTh1 benchmark and a real-world, previously unseen dataset: Kaggle's
Electric Motor Temperature (PMSM rotor temperature) dataset. On ETTh1 we
close part of the gap to the paper's reported numbers through RevIN and
tuning, but remain roughly 8x off Table 1, plausibly dominated by backbone
size (124M vs. the paper's 7B-parameter Llama backbone). On the real-world
dataset, the reprogrammed model clearly outperforms a naive persistence
baseline (41-49% lower error), demonstrating genuine generalization to
unseen data — but is in turn outperformed by a small, from-scratch LSTM
baseline trained on the same data and windows (31% lower MSE, 22% lower
MAE, ~26x fewer parameters, ~5x less training time). We report this as an
honest negative result and discuss what it does and doesn't imply about
the reprogramming approach.

## 1. Motivation

Forecasting motor temperature from sensor data is a well-studied problem
usually tackled with purpose-built sequence models (LSTMs, TCNs,
Transformers trained from scratch on the target domain). Time-LLM
proposes an alternative: keep a large pretrained language model frozen,
and train only a lightweight "reprogramming" layer that translates
numeric time-series patches into something resembling the LLM's native
token-embedding space. The appeal is that the frozen LLM contributes
general sequence-reasoning ability learned from massive pretraining,
without the cost of training a large model from scratch or fine-tuning
it.

This project asks a practical question relevant to anyone considering
this approach for a real sensor-forecasting task: **on modest hardware,
with a small frozen backbone, does reprogramming actually beat a simple
task-specific baseline you could train in an afternoon?** We answer this
empirically rather than assuming the paper's headline results (obtained
with a 7B-parameter backbone and full training budget) transfer to a
constrained, CPU-only, GPT-2-scale setting.

## 2. Method

Time-LLM's core mechanism, as implemented here:

1. **Patch embedding.** The input window is split into overlapping
   patches (`patch_len=16`, `stride=8`) and linearly projected to a small
   embedding dimension `d_model`.
2. **Patch reprogramming.** A multi-head cross-attention layer lets each
   patch embedding attend over a fixed subsample of the frozen LLM's own
   vocabulary embedding matrix ("text prototypes"), producing an
   embedding in the LLM's native space without ever editing the LLM's
   weights.
3. **RevIN (reversible instance normalization).** Each input window is
   normalized using its own mean/std (not a global training-set
   statistic), and the model's output is denormalized with the same
   per-window statistics before the loss is computed. This is the
   mechanism the paper uses to handle distribution shift between windows.
4. **Prompt-as-Prefix (PaP).** A short text prompt — task instruction
   plus per-window statistics (min, max, median, trend, top autocorrelation
   lags) — is tokenized, embedded via the frozen LLM's own embedding
   table, and prepended to the reprogrammed patch sequence before the
   frozen LLM forward pass. Left-padding and recomputed position IDs keep
   variable-length prompts aligned. The frozen LLM's output positions
   corresponding to the prefix are discarded; only the patch positions
   feed the final forecast head.
5. **Output projection.** The frozen LLM's output over the patch
   positions is flattened and linearly projected to the forecast horizon.

Only the patch embedding, reprogramming layer, RevIN's two affine
parameters, and the output projection are trained. The LLM itself never
updates — in our runs, 0.07-1.12% of total parameters were trainable,
consistent with the paper's efficiency claims.

**Deviations from the paper**, made for tractability on CPU-only, no-GPU
hardware:
- Backbone: GPT-2 (124M) instead of the paper's primary Llama-7B.
- PaP prompt: compact (instruction + numeric stats only) rather than the
  paper's richer natural-language dataset description and domain
  knowledge, since every prompt token adds sequence length the frozen LLM
  must process on every forward pass.
- Smaller `d_model` / prototype counts and shorter training schedules
  than the paper's tuned, per-dataset configurations.

## 3. Experimental Setup

All experiments ran on CPU only (no GPU), Python 3.11, PyTorch/Transformers
CPU builds. Two datasets:

- **ETTh1** (paper's own benchmark; Electricity Transformer Temperature,
  hourly): standard 96→24 forecasting window, chronological 70/10/20
  train/val/test split.
- **Kaggle Electric Motor Temperature** (wkirgsn/electric-motor-temperature,
  Paderborn University): 185 hours of PMSM test-bench recordings at 2Hz.
  The raw file concatenates 69 independent measurement sessions
  (`profile_id`) with no timestamp column; sliding windows across session
  boundaries would silently splice unrelated recordings together, so we
  filtered to a single session (`profile_id=70`, 25,677 rows, ≈3.6 hours)
  before windowing. Target: `pm` (permanent-magnet/rotor temperature — the
  classic hard-to-measure-directly quantity in this literature). Window
  size `seq_len=240`/`pred_len=60` (2 minutes of history → 30 seconds
  ahead) was chosen deliberately over the ETTh1-matched 96/24 because at
  2Hz sampling, a 12-second-ahead forecast is close to a trivial
  persistence task for a slowly-varying physical quantity — we verified
  this empirically (Section 4.3) before committing compute to the longer
  run.

We also implemented a from-scratch **LSTM baseline** (`hidden_size=64`,
`num_layers=2`, same RevIN normalization, same window/split code) to give
the real-world comparison a fair, cheap-to-train reference point beyond
naive persistence.

## 4. Results

### 4.1 Pipeline correctness (synthetic data)

A 3-epoch sanity run on synthetic motor-temperature data confirmed the
full pipeline — real GPT-2 weights loading and staying frozen, forward/
backward passes, checkpointing — works end-to-end, with loss converging
monotonically (train MSE 1.60 → 0.18 across 3 epochs).

### 4.2 ETTh1 (paper's benchmark)

| Configuration | Trainable params | Test MSE | Test MAE |
|---|---|---|---|
| Global (train-set) normalization | 240,984 (0.19%) | 5.9605 | 1.9694 |
| + RevIN | 240,986 (0.19%) | 3.3831 | 1.3817 |
| **+ capacity (`d_model` 16→32) + LR schedule** | 278,938 (0.22%) | **3.2489** | **1.3155** |
| + Prompt-as-Prefix (6 epochs, time-limited) | 278,938 (0.22%) | 3.4387 | 1.3793 |
| Paper (Time-LLM, Llama-7B, Table 1) | — | ~0.36–0.40 | ~0.40–0.41 |

RevIN produced the single largest improvement (−43% MSE, −30% MAE over
global normalization) — the clearest evidence that per-window
normalization, not just tuning, matters for this task. Doubling model
capacity and adding an LR scheduler yielded only a further −4% MSE,
suggesting diminishing returns from tuning within this architecture.
Prompt-as-Prefix, evaluated under a reduced 6-epoch budget (a full
15-epoch PaP run was estimated at ~13 hours of CPU time and not
completed), came out marginally worse than the non-PaP configuration —
inconclusive given the training-budget confound and the deliberately
compact prompt (Section 5).

We remain roughly 8x off the paper's reported MSE. The most plausible
explanation is backbone capacity: our frozen GPT-2 has 124M parameters
versus the paper's primary 7B-parameter Llama backbone — over 50x fewer
parameters for the reprogrammed patches to draw structure from.

### 4.3 Real-world validation: window-size sensitivity

Before committing several CPU-hours to a benchmark run, we ran a quick
3-epoch check at the ETTh1-matched window size (`seq_len=48`,
`pred_len=12`, i.e. 6 seconds ahead at 2Hz): test MSE=0.0097, MAE=0.0371.
This near-zero error is not evidence of a strong model — it reflects that
motor temperature barely changes in 6 seconds, making the task close to
trivial persistence. We used this check to justify the longer,
non-trivial `seq_len=240`/`pred_len=60` window for the real benchmark.

### 4.4 Real-world benchmark and baseline comparison

| Method | Test MSE | Test MAE | Trainable params | Train time (CPU) |
|---|---|---|---|---|
| Naive persistence (repeat last value) | 0.0460 | 0.1625 | 0 | — |
| Time-LLM (frozen GPT-2, reprogrammed, 15 epochs) | 0.0236 | 0.0954 | 1,412,542 | ~7.97h |
| **LSTM (from scratch, 30 epochs, best at epoch 4)** | **0.0162** | **0.0745** | **54,334** | **~1.55h** |

Both learned methods convincingly beat naive persistence (Time-LLM: −49%
MSE / −41% MAE), confirming genuine generalization to a dataset the
architecture had never seen and that isn't one of the paper's own
benchmarks. However, **the from-scratch LSTM baseline outperforms the
reprogrammed Time-LLM model outright**: 31% lower MSE, 22% lower MAE,
using ~26x fewer trainable parameters and ~5x less wall-clock training
time, with no frozen 124M-parameter backbone required at all. The LSTM
did show clear overfitting past epoch 4 (train MSE kept falling to 0.067
while val MSE rose to 0.28 despite eight LR reductions), and its best
checkpoint (epoch 4, reached in well under 15 minutes) was what was
evaluated.

## 5. Discussion

**RevIN is not optional.** Across every experiment, switching from
global to per-window normalization was the single highest-leverage
change we made — more impactful than doubling model capacity, adding a
learning-rate schedule, or adding Prompt-as-Prefix combined.

**Prompt-as-Prefix's contribution here is genuinely unresolved, not
negative.** We verified the mechanism is implemented correctly (left-
padding, recomputed position IDs for variable-length prompts, correct
masking of prefix vs. patch positions in the output) and functions
end-to-end. But CPU constraints forced two confounds in the only
head-to-head test we ran: a much smaller training budget than the
comparison run, and a deliberately stripped-down prompt lacking the rich
natural-language dataset context the paper credits with giving the LLM
something to reason about. A fair test needs either a GPU or substantially
more CPU time than was available for this project.

**The GPT-2-vs-LSTM result is the most interesting finding of this
project, and it cuts against the reprogramming approach on this specific
task.** It should not be read as "Time-LLM doesn't work" — it clearly
does better than a naive baseline, and the paper's claims are principally
about a 7B-parameter backbone evaluated across eight diverse benchmark
datasets, with particular emphasis on few-shot and zero-shot transfer.
What it does suggest is a **scope condition**: on a single, data-abundant,
smoothly-varying real sensor series, with a full training budget
available and a modest (124M-parameter) frozen backbone, the overhead of
routing through a general-purpose frozen LLM did not translate into an
accuracy advantage over a small, fast, purpose-built recurrent model.
The value proposition of reprogramming — leveraging broad pretrained
knowledge — plausibly matters most exactly where our setup is weakest:
backbone scale, and few-shot/cross-domain settings rather than
abundant single-domain data. This is a useful, citable scope boundary
rather than a refutation.

**Compute cost is a first-class result, not a footnote.** The LSTM
reached its best checkpoint in under 15 minutes; the Time-LLM real-data
run took ~8 hours for a comparable (in fact better) fit. For any
practitioner choosing a forecasting method under a compute budget, this
tradeoff is directly decision-relevant and arguably more actionable than
the accuracy numbers alone.

## 6. Limitations and Future Work

- **Backbone scale** is the most likely lever left untested: repeating
  the ETTh1 and real-data experiments with a 7B-parameter backbone (as in
  the original paper) on a GPU would directly test whether the
  LSTM-beats-Time-LLM result is an artifact of backbone size rather than
  the reprogramming idea itself.
- **A fair, full-budget Prompt-as-Prefix comparison** (matched epochs,
  full natural-language prompt) is needed before drawing conclusions
  about PaP's contribution.
- **A single real-world session** (`profile_id=70`) was used; the dataset
  has 69 sessions with varying operating conditions, and testing
  generalization across sessions (not just within one) would be a
  stronger real-world claim.
- **The LSTM baseline is intentionally simple** (2-layer, 64 hidden
  units); a more competitive baseline (e.g. a small Transformer, or a
  tuned/regularized LSTM with dropout scheduling to reduce the overfitting
  observed after epoch 4) would sharpen the comparison further.

## 7. Reproducibility

All code, exact hyperparameters, and per-epoch logs for every run
reported here are in the accompanying repository's `README.md`, along
with setup instructions, the Kaggle dataset's session-splitting caveat,
and exact commands to reproduce every table in this report:

- `model.py` — Time-LLM model (RevIN, Prompt-as-Prefix, patch
  reprogramming, frozen GPT-2 backbone)
- `lstm_baseline.py` / `train_lstm.py` — the from-scratch LSTM baseline
- `train.py`, `data_provider.py` — training loop and dataset handling
- `README.md` — full experimental log with every run's per-epoch metrics

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
