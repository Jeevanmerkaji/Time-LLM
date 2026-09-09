# Time-LLM reproduction — project handoff

Read this first in any new session on this project. Full background,
method description, and all historical results are in `README.md` and
`REPORT.md` — read those next for context, this file only covers
**current state and what's left to do**.

## What this project is

A from-scratch reproduction of Time-LLM (Jin et al., ICLR 2024) — forecasting
motor temperature by reprogramming a frozen GPT-2 — plus honest baseline
comparisons (LSTM, DLinear) on a real Kaggle motor-temperature dataset. Part
of a PhD application bridge narrative (sensor/LSTM background → LLM methods).

## Current state (as of 2026-09-09, last real activity 2026-08-24)

The main line of work (single-session Step 1-4 results in `README.md`/
`REPORT.md`) is **done and written up**.

The active, unfinished piece is a **"full rigor" experiment matrix** for
strengthening the arXiv writeup: 3 methods (`dlinear`, `lstm`, `time_llm`) x
3 real Kaggle sessions (`profile_id` 70, 62, 44) x 3 seeds = 27 runs, driven
by `run_full_rigor.py`.

**22 of 27 runs are complete** (see `results_full_rigor.jsonl`, one JSON
line per finished run — `status: "ok"` means it counted; a `failed` or
`unparsed` entry for the same combo just means it was retried). **5 runs
remain**:
- profile 44, lstm, seed 1
- profile 44, lstm, seed 2
- profile 44, time_llm, seed 0
- profile 44, time_llm, seed 1
- profile 44, time_llm, seed 2

## How to resume

```bash
py -3.11 -m venv venv
./venv/Scripts/pip install torch transformers pandas numpy   # Windows
python run_full_rigor.py
```

`run_full_rigor.py` reads `results_full_rigor.jsonl` and automatically skips
any `(profile_id, method, seed)` combo already marked `"ok"` — just re-run
it, no flags needed. It runs sequentially (CPU-bound, ~20-40 min for
dlinear/lstm runs, several hours for each time_llm run — the remaining 3
time_llm runs are the long pole, likely 5-9 hours each on CPU based on prior
runs of the same config). Expect the whole remaining batch to take roughly a
day of wall-clock time on CPU.

Once all 27 rows show `status: "ok"` in `results_full_rigor.jsonl`, the
matrix is done — next step is aggregating mean/std per (profile, method)
across seeds for the arXiv writeup (not yet started; no aggregation script
exists yet, would need to be written).

## Gotchas learned the hard way

- `checkpoints/full_rigor/lstm_p44_s1.pt.resume` is a leftover in-progress
  checkpoint from an interrupted run — harmless, `train_lstm.py` will
  overwrite/recreate it when that run is redone.
- One `time_llm p62 seed0` run failed once (HF Hub rate-limit warning, not
  fatal) before succeeding on retry — if a run fails, just re-run
  `run_full_rigor.py` again, it retries anything not marked `"ok"`.
- Python must be ≤3.11 (no PyTorch wheels for 3.14 as of this writing) —
  use `py -3.11` explicitly on Windows if the default `python` is newer.
- The raw Kaggle file `data/electric_motor_temp/measures_v2.csv` (300MB) is
  NOT filtered by profile — always use the pre-filtered
  `data/electric_motor_temp_profile{44,62,70}.csv` files for training,
  never the raw file directly (see README Step 3 for why).

## Not yet in git

This project has no git repository yet (working directly on the filesystem).
If you want version history / GitHub going forward, that's a separate ask —
this handoff was written for a full folder-copy transfer (USB/cloud), not a
git-based one.
