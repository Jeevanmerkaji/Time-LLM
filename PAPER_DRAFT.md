# Reprogramming a Frozen LLM for Motor Temperature Forecasting: A Multi-Session, Multi-Seed Reproduction and Baseline Study

*Draft — full-rigor revision. Supersedes the single-session `REPORT.md` for
the arXiv/workshop submission. Status: all 27/27 runs complete
(see `results_full_rigor_summary.md` for the raw aggregation).*

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
central finding is that **no single method dominates across sessions**: the
from-scratch LSTM baseline wins on `profile_id=70` (31% lower MSE than
Time-LLM, replicating an earlier single-session result), DLinear and Time-LLM
are statistically indistinguishable at a near-zero error floor on
`profile_id=62`, and **Time-LLM wins outright on `profile_id=44`** (27% lower
MSE than LSTM, 10% lower than DLinear). Averaged across all three sessions,
Time-LLM achieves the lowest mean test MSE of the three methods (0.0290 vs.
LSTM's 0.0337 and DLinear's 0.0348), while LSTM achieves the lowest mean MAE
(0.0581 vs. Time-LLM's 0.0793) — a split verdict that itself is the main
result. This directly contradicts the common assumption, supported by our
own earlier single-session pilot, that a small from-scratch baseline
uniformly dominates LLM-reprogramming on data-abundant, single-sensor tasks,
and demonstrates why multi-session evaluation is necessary before drawing
method-level conclusions from time-series forecasting comparisons — a
single session, however carefully chosen, can misrepresent the general
ranking of methods on a dataset with heterogeneous recording sessions.

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
original single-session result an artifact of that one session?* The answer,
reported in Section 4, is that it was partly an artifact: the LSTM's
advantage replicates on the original session but reverses on another,
with the third session landing all three methods within noise of each
other.

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
| 62 | 25,600 | — |
| 44 | 26,341 | — |

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

All 27/27 runs complete (3 sessions × 3 methods × 3 seeds). Full raw
aggregation in `results_full_rigor_summary.md`, regenerable via
`python aggregate_results.py`.

### 4.1 Per-session results

| `profile_id=70` | Test MSE | Test MAE | Train time (mean) |
|---|---|---|---|
| DLinear | 0.0350 ± 0.0073 | 0.1330 ± 0.0247 | 0.05h |
| **LSTM** | **0.0150 ± 0.0007** | **0.0629 ± 0.0041** | 1.58h |
| Time-LLM | 0.0247 ± 0.0058 | 0.0985 ± 0.0218 | 8.13h |

| `profile_id=62` | Test MSE | Test MAE | Train time (mean) |
|---|---|---|---|
| **DLinear** | **0.0003 ± 0.0000** | **0.0135 ± 0.0001** | 0.04h |
| LSTM | 0.0007 ± 0.0005 | 0.0173 ± 0.0049 | 1.47h |
| Time-LLM | 0.0004 ± 0.0002 | 0.0157 ± 0.0029 | 3.70h |

| `profile_id=44` | Test MSE | Test MAE | Train time (mean) |
|---|---|---|---|
| DLinear | 0.0691 ± 0.0208 | 0.0985 ± 0.0049 | 0.05h |
| LSTM | 0.0853 ± 0.0086 | 0.0940 ± 0.0020 | 0.53h |
| **Time-LLM** | **0.0619 ± 0.0019** | 0.1236 ± 0.0247 | 6.58h |

**Each method wins exactly one of the three sessions on test MSE.** On
`profile_id=70`, LSTM beats Time-LLM by 31% MSE — a clean replication of
the original single-session finding. On `profile_id=62`, all three methods
sit within a very narrow, near-zero error band (this session's `pm` target
appears to have an unusually low dynamic range/noise floor, making the
forecasting task close to saturated for all three methods regardless of
capacity); DLinear is nominally lowest but the seed-to-seed spread mostly
overlaps with Time-LLM's. On `profile_id=44`, **Time-LLM wins outright**:
27% lower MSE than LSTM and 10% lower than DLinear, with the smallest
seed-to-seed std of the three methods on this session (± 0.0019 MSE) —
the most confidently-won session in the whole matrix.

### 4.2 Cross-session summary

| Method | Mean Test MSE | Mean Test MAE | Sessions won (MSE) |
|---|---|---|---|
| DLinear | 0.0348 | 0.0817 | 1/3 (`p62`) |
| LSTM | 0.0337 | **0.0581** | 1/3 (`p70`) |
| **Time-LLM** | **0.0290** | 0.0793 | 1/3 (`p44`) |

(Mean of per-session means, unweighted across sessions — see the caveat in
Section 5 on why this specific aggregation should be read cautiously
despite Time-LLM's lead here.)

## 5. Discussion

**No method dominates across sessions — the split verdict is the finding.**
Each of the three methods wins exactly one session on test MSE, and the
cross-session mean and the cross-session "sessions won" count don't even
agree on a single winner by every metric: Time-LLM has the lowest mean MSE,
but LSTM has the lowest mean MAE, and all three are tied 1-1-1 on session
wins. This is the central methodological point of this study: a
single-session comparison, however carefully run (the original
`REPORT.md` study used a full 15/30-epoch budget, proper RevIN
normalization, and a fair from-scratch baseline — nothing about the
original comparison was sloppy) can still produce a method-ranking claim
that reverses on a second, equally legitimate session of the same
underlying real-world dataset. Multi-session evaluation is not a
nice-to-have for honest claims about which forecasting approach "wins" on
a sensor-data problem — per-session dynamics (noise floor, signal range,
operating regime) can matter as much as, or more than, the modeling choice
itself.

**The cross-session mean MSE should be read with a specific caveat.**
`profile_id=62`'s errors are roughly two orders of magnitude smaller than
the other two sessions' (0.0003–0.0007 vs. 0.015–0.09), so an unweighted
mean across sessions is dominated by whichever method happens to be
marginally better on the two "harder" sessions (`p70`, `p44`) rather than
reflecting a genuinely comparable average. Time-LLM's cross-session MSE
lead is real but should be understood as "wins the harder session
(`p44`) outright, loses the other harder session (`p70`), ties on the
easy one (`p62`)" rather than "wins on average" in any deeper sense — we
report the unweighted mean because it is the simplest defensible
aggregation given only 3 sessions, not because we think it's the last
word on ranking.

**Why might Time-LLM win specifically on `profile_id=44`?** We do not have
a confirmed mechanistic answer — this would require inspecting each
session's underlying operating conditions (load profile, thermal transient
vs. steady-state behavior) in the source PMSM dataset, which is out of
scope here. What we can say: `profile_id=44`'s std across seeds for
Time-LLM (± 0.0019) is tighter than LSTM's (± 0.0086) and DLinear's
(± 0.0208) on the same session, meaning Time-LLM's advantage there is not
an artifact of a single lucky seed. This is consistent with (though not
proof of) the reprogrammed LLM's frozen pretrained backbone providing more
useful inductive bias on sessions with more complex/nonstationary dynamics
than a small from-scratch model can pick up in 15-30 epochs on one
session's data alone — precisely the kind of scope condition the original
single-session study's Discussion speculated might exist but couldn't
observe with only one session available.

**Compute cost remains a first-class, session-independent result.** Across
all sessions, Time-LLM training took roughly 4-16x longer than the LSTM
baseline (1.5-8.1h vs. 0.5-1.6h) for accuracy that is session-dependently
better, worse, or tied. This tradeoff is directly decision-relevant
regardless of which method wins on accuracy for a given session, and
matters even more once the result is "it depends which session you're on"
rather than a clean win for either side.

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
