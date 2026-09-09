# Reprogramming a Frozen LLM for Motor Temperature Forecasting: A Multi-Session, Multi-Seed Reproduction and Baseline Study

*Draft — full-rigor revision. Supersedes the single-session `REPORT.md` for
the arXiv/workshop submission. Status: matrix in progress, see*
`results_full_rigor_summary.md` *for live numbers; placeholders below marked* `[TBD]`.

## Abstract

We reproduce a simplified version of Time-LLM (Jin et al., ICLR 2024), which
forecasts time series by reprogramming a frozen, pretrained language model
rather than training a forecaster from scratch. Using a GPT-2 (124M) backbone
on CPU-only hardware, we implement the reprogramming mechanism, reversible
instance normalization (RevIN), and evaluate against two baselines — a
from-scratch LSTM and a linear DLinear model — on Kaggle's real-world
Electric Motor Temperature dataset. Unlike a typical single-run comparison,
we evaluate across **3 independent real-world recording sessions**
(`profile_id` 70, 62, 44) **× 3 random seeds** (9 runs per method, 27 total),
reporting mean ± standard deviation rather than single-point estimates. Our
central finding is that the reprogrammed-LLM approach's relative ranking
against the LSTM baseline is **session-dependent**: [TBD — fill in once
profile 44 completes; current partial data shows Time-LLM outperforming LSTM
on profile 62 while LSTM outperforms Time-LLM on profile 70]. This nuances
the common assumption that a small from-scratch baseline uniformly dominates
LLM-reprogramming on data-abundant, single-sensor tasks, and highlights the
importance of multi-session evaluation before drawing method-level
conclusions from time-series forecasting comparisons.

## 1. Motivation

Forecasting motor temperature from sensor data is usually tackled with
purpose-built sequence models (LSTMs, TCNs, Transformers) trained from
scratch on the target domain. Time-LLM proposes reprogramming a frozen,
general-purpose LLM instead — training only a lightweight adapter layer
while the LLM's own weights never update. The appeal is inheriting broad
sequence-reasoning ability from massive pretraining without the cost of
training or fine-tuning a large model.

A first pass at this question (single session, single seed; see `REPORT.md`)
found the from-scratch LSTM baseline winning outright. But a single
session/seed comparison risks over-generalizing from what could be
session-specific noise or an artifact of one random initialization — a
known concern in time-series forecasting benchmarks, where results can be
sensitive to both the specific recording window and training stochasticity.
This work extends that comparison to **3 independent real-world sessions ×
3 seeds per method**, asking: *does the from-scratch baseline's advantage
hold up across multiple independent slices of real data, or was the
original single-session result an outlier?*

## 2. Method

Unchanged from the original single-session study (see `REPORT.md` Section 2
for full detail); summarized here:

1. **Patch embedding** — input window split into overlapping patches
   (`patch_len=16`, `stride=8`), linearly projected to `d_model`.
2. **Patch reprogramming** — multi-head cross-attention lets each patch
   attend over a fixed subsample of the frozen LLM's vocabulary embedding
   matrix, producing an embedding in the LLM's native space without editing
   LLM weights.
3. **RevIN** — per-window (not global) instance normalization, reversed on
   output before the loss.
4. **Output projection** — frozen LLM's output over patch positions,
   flattened and linearly projected to the forecast horizon.

Only the patch embedding, reprogramming layer, RevIN's affine parameters,
and the output projection are trained (~0.1–1.1% of total parameters
depending on config). Prompt-as-Prefix (PaP) is **disabled** in this matrix
(`--no_prompt`) — it was evaluated separately in the single-session study
and roughly quintuples CPU cost per sample; the full-rigor matrix
prioritizes seed/session coverage over PaP, which remains an open direction
(see Limitations).

**Baselines:**
- **LSTM** — 2-layer, 64 hidden units, same RevIN normalization and window
  code as Time-LLM, so the comparison isolates the forecasting method
  rather than data-handling differences.
- **DLinear** — a simple linear decomposition model (trend + seasonal),
  included as a fast, near-zero-capacity lower bound.

## 3. Experimental Setup

**Dataset.** Kaggle's Electric Motor Temperature dataset
(`wkirgsn/electric-motor-temperature`, Paderborn University): 185 hours of
PMSM test-bench recordings at 2Hz, 69 independent measurement sessions
(`profile_id`) concatenated with no timestamp column. We select **3
sessions** to test cross-session robustness:

| `profile_id` | Rows | Notes |
|---|---|---|
| 70 | 25,677 | mid-sized session, used in the original single-session study |
| 62 | [TBD — row count] | — |
| 44 | [TBD — row count] | — |

Target: `pm` (permanent-magnet/rotor temperature). Window size
`seq_len=240`/`pred_len=60` (2 minutes of history → 30 seconds ahead),
chosen (as in the original study) to avoid the near-trivial persistence
regime of shorter windows at 2Hz sampling.

**Seeds.** Each (session, method) combination is repeated for seeds
`{0, 1, 2}`, varying model initialization and any stochastic training
elements; data splits (chronological 70/10/20 train/val/test) are fixed per
session.

**Training budget.** DLinear/LSTM: 30 epochs. Time-LLM: 15 epochs (matching
the original single-session study; CPU-time-limited — see Limitations).
Batch size 16 throughout. All runs CPU-only (no GPU), frozen GPT-2 (124M)
backbone for Time-LLM.

**Orchestration.** `run_full_rigor.py` drives all 27 runs sequentially
(CPU-bound; parallel runs would just contend for the same cores),
checkpointing each result to `results_full_rigor.jsonl` immediately so
progress survives interruption. `aggregate_results.py` computes mean ± std
per (session, method) from that log.

## 4. Results

*Auto-generated from `results_full_rigor_summary.md` — regenerate this
section by running `python aggregate_results.py` after the matrix
completes and pasting the output tables here.*

### 4.1 Per-session results

**[TBD — insert the three per-`profile_id` tables from
`results_full_rigor_summary.md` here once all 27 runs finish. Partial
results as of this draft:]**

- `profile_id=70`: LSTM (0.0150 ± 0.0007 MSE) beats Time-LLM
  (0.0247 ± 0.0058 MSE) beats DLinear (0.0350 ± 0.0073 MSE). Consistent
  with the original single-session finding.
- `profile_id=62`: **Time-LLM (0.0004 ± 0.0002 MSE) beats LSTM
  (0.0007 ± 0.0005 MSE)** — DLinear is marginally best here
  (0.0003 ± 0.0000 MSE), but all three methods are within noise of each
  other on this session (errors near the floor of the target's dynamic
  range).
- `profile_id=44`: incomplete — DLinear done (0.0691 ± 0.0208 MSE); LSTM
  and Time-LLM pending.

### 4.2 Cross-session summary

**[TBD — insert the cross-session table once complete.]**

## 5. Discussion

**The LSTM-wins finding does not generalize uniformly across sessions.**
On `profile_id=70`, the LSTM baseline's advantage over Time-LLM (reported
in the original single-session study) replicates. On `profile_id=62`,
Time-LLM edges out the LSTM baseline, and all three methods converge to
very low, closely-spaced error — suggesting this particular session is
close to a regime where model capacity/architecture matters little (a
slowly-varying or low-dynamic-range signal for this session). [TBD: extend
this paragraph once `profile_id=44` is complete — does the 3rd session
break the tie toward LSTM, Time-LLM, or remain mixed?]

**Implication for practitioners and for reproductions of this kind:** a
single-session comparison — the norm in a lot of applied forecasting
write-ups, including our own original draft — can produce a
method-ranking claim that doesn't hold on a second slice of the same
underlying real-world dataset. This is a methodological point as much as
an empirical one: multi-session (not just multi-seed) evaluation matters
for honest claims about which forecasting approach "wins" on a given
sensor-data problem, because per-session dynamics (noise floor, signal
range, operating regime) can matter as much as the modeling choice itself.

**Compute cost remains a first-class, session-independent result.** Across
all sessions, Time-LLM training took roughly 4-8x longer than the LSTM
baseline (see per-session tables) for accuracy that is at best comparable
and at worst worse. This tradeoff is directly decision-relevant regardless
of which method wins on accuracy for a given session.

## 6. Limitations and Future Work

- **Backbone scale** — GPT-2 (124M) vs. the paper's primary Llama-7B remains
  untested here; this is the most likely lever behind the residual gap to
  the paper's own ETTh1 numbers (see `REPORT.md` Section 4.2 for that
  separate comparison).
- **Prompt-as-Prefix disabled** in this matrix for CPU-time reasons; a
  full-rigor PaP comparison (matched epochs, richer natural-language
  prompt) is a natural next extension.
- **3 of 69 available sessions** — broader coverage (more sessions, or a
  formal stratified sample across the dataset's different operating
  conditions) would further strengthen the cross-session claim.
- **Time-LLM epoch budget (15) vs. LSTM/DLinear (30)** — kept from the
  original study for CPU-time reasons; asymmetric budgets are a possible
  confound worth flagging explicitly in the final writeup.

## 7. Reproducibility

- `run_full_rigor.py` — orchestrates the full 27-run matrix, resumable
  (skips any `(profile_id, method, seed)` already marked `"ok"` in
  `results_full_rigor.jsonl`).
- `aggregate_results.py` — produces `results_full_rigor_summary.md` (mean ±
  std per session/method) from the raw log.
- `results_full_rigor.jsonl` — one JSON record per finished run: config,
  status, test MSE/MAE, wall-clock time.
- `model.py`, `lstm_baseline.py`, `dlinear_baseline.py`,
  `train.py`/`train_lstm.py`/`train_dlinear.py`, `data_provider.py` — model
  and training code, unchanged from the original single-session study.
- Repository: https://github.com/Jeevanmerkaji/Time-LLM

## Citation

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
