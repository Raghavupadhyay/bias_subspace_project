#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "scikit-learn",
#   "tqdm",
#   "transformer-lens",
#   "sae-lens",
#   "pyyaml",
#   "pytest"
# ]
# ///

import os
import subprocess


def run_step(name, cmd):
    print(f"\n========== {name} ==========")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {name}")


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # ✅ FIX: add PYTHONPATH
    run_step("Running tests", "PYTHONPATH=. pytest tests/")

    run_step("Baseline probe", "PYTHONPATH=. python experiments/exp01_baseline_probe.py")
    run_step("SAE features", "PYTHONPATH=. python experiments/exp02_sae_features.py")
    run_step("Cancellation pairs", "PYTHONPATH=. python experiments/exp03_cancellation_pairs.py")
    run_step("Causal ablation", "PYTHONPATH=. python experiments/exp04_causal_ablation.py")

    print("\n✅ PIPELINE COMPLETE")


if __name__ == "__main__":
    main()