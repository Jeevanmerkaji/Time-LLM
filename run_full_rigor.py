"""
Orchestrator for the "full rigor" experiment matrix: 3 methods x 3 real-world
sessions x 3 seeds = 27 runs, for the arXiv-strengthening pass.

Runs sequentially (CPU-bound single-machine constraint -- parallel runs would
just contend for the same cores). Each run's result is appended to
results_full_rigor.jsonl immediately after it finishes, so progress survives
a crash or interruption partway through; rerunning this script skips any
(session, method, seed) combination already present in the log.

Usage:
    python run_full_rigor.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
LOG_PATH = Path("results_full_rigor.jsonl")
CKPT_DIR = Path("checkpoints/full_rigor")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [70, 62, 44]
SEEDS = [0, 1, 2]
SEQ_LEN, PRED_LEN = 240, 60

# (method_name, script, extra_args, epochs)
METHODS = [
    ("dlinear", "train_dlinear.py", [], 30),
    ("lstm", "train_lstm.py", [], 30),
    ("time_llm", "train.py", ["--d_model", "32", "--num_prototypes", "200", "--no_prompt"], 15),
]

RESULT_RE = re.compile(r"FINAL TEST RESULTS: MSE=([\d.]+)\s+MAE=([\d.]+)")


def already_done():
    """Only status=='ok' counts as done -- failed/timeout entries stay in the
    log for the record, but the combo gets retried on the next run."""
    done = set()
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add((rec["profile_id"], rec["method"], rec["seed"]))
    return done


def run_one(profile_id, method, script, extra_args, epochs, seed):
    csv_path = f"data/electric_motor_temp_profile{profile_id}.csv"
    save_path = str(CKPT_DIR / f"{method}_p{profile_id}_s{seed}.pt")
    cmd = [
        PYTHON, script,
        "--csv", csv_path, "--target", "pm",
        "--seq_len", str(SEQ_LEN), "--pred_len", str(PRED_LEN),
        "--epochs", str(epochs), "--batch_size", "16",
        "--seed", str(seed), "--save_path", save_path,
    ] + extra_args

    print(f"\n=== profile_id={profile_id} method={method} seed={seed} ===", flush=True)
    print(" ".join(cmd), flush=True)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=43200)  # 12h safety cap
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"!! TIMED OUT after {elapsed:.0f}s", flush=True)
        return {"profile_id": profile_id, "method": method, "seed": seed,
                "status": "timeout", "elapsed_s": elapsed}

    elapsed = time.time() - t0
    stdout = proc.stdout
    print(stdout[-2000:], flush=True)  # tail of this run's log, for visibility
    if proc.returncode != 0:
        print(f"!! FAILED (exit {proc.returncode})", flush=True)
        print(proc.stderr[-2000:], flush=True)
        return {"profile_id": profile_id, "method": method, "seed": seed,
                "status": "failed", "elapsed_s": elapsed,
                "stderr_tail": proc.stderr[-2000:]}

    m = RESULT_RE.search(stdout)
    if not m:
        print("!! Could not parse FINAL TEST RESULTS from stdout", flush=True)
        return {"profile_id": profile_id, "method": method, "seed": seed,
                "status": "unparsed", "elapsed_s": elapsed}

    mse, mae = float(m.group(1)), float(m.group(2))
    print(f"-> MSE={mse:.4f} MAE={mae:.4f} ({elapsed:.0f}s)", flush=True)
    return {"profile_id": profile_id, "method": method, "seed": seed,
            "status": "ok", "test_mse": mse, "test_mae": mae, "elapsed_s": elapsed}


def main():
    done = already_done()
    total = len(SESSIONS) * len(METHODS) * len(SEEDS)
    i = 0
    for profile_id in SESSIONS:
        for method, script, extra_args, epochs in METHODS:
            for seed in SEEDS:
                i += 1
                if (profile_id, method, seed) in done:
                    print(f"[{i}/{total}] skip (already done): p{profile_id} {method} seed{seed}")
                    continue
                print(f"[{i}/{total}] running: p{profile_id} {method} seed{seed}")
                result = run_one(profile_id, method, script, extra_args, epochs, seed)
                with LOG_PATH.open("a") as f:
                    f.write(json.dumps(result) + "\n")

    print("\nAll runs complete (or already were). See results_full_rigor.jsonl")


if __name__ == "__main__":
    main()
