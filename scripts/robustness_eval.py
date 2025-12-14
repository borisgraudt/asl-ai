"""
Robustness evaluation: repeat training multiple times with different random seeds.

This measures *training stability* (initialization / stochasticity) on a fixed dataset split.
It writes:
- `logs/robustness_summary.json`
- per-run histories in `logs/` (via train_model run_name)
"""

import json
import os
import statistics as stats
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.train import train_model
from src.utils.config import config
from src.utils.logger import setup_logger, get_logger


def _aggregate(values):
    return {
        "mean": float(stats.mean(values)) if values else None,
        "stdev": float(stats.pstdev(values)) if len(values) > 1 else 0.0,
        "runs": [float(v) for v in values],
    }


def run_arch(arch: str, seeds: list[int]) -> dict:
    results = []
    for seed in seeds:
        config.TRAINING_CONFIG["architecture"] = arch
        run_name = f"robust_{arch}_seed{seed}"
        model_path = config.MODELS_DIR / f"{run_name}.h5"
        res = train_model(model_path=model_path, save_plots=False, run_name=run_name, seed=seed)
        results.append(res)
    accs = [r["test_accuracy"] for r in results]
    top3s = [r["test_top3_accuracy"] for r in results]
    losses = [r["test_loss"] for r in results]
    times = [r["training_time_minutes"] for r in results]
    return {
        "seeds": seeds,
        "metrics": {
            "test_accuracy": _aggregate(accs),
            "test_top3_accuracy": _aggregate(top3s),
            "test_loss": _aggregate(losses),
            "training_time_minutes": _aggregate(times),
        },
    }


if __name__ == "__main__":
    setup_logger("asl_ai.robustness", log_level=20)
    logger = get_logger(__name__)

    # Default to 3 runs (fast enough to be practical)
    seeds_env = os.getenv("ASL_AI_ROBUST_SEEDS", "1,2,3")
    seeds = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]

    logger.info(f"Running robustness eval with seeds={seeds}")

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": {
            "processed_dir": str(config.PROCESSED_DATA_DIR),
        },
        "mlp": run_arch("mlp", seeds),
        "moe": run_arch("moe", seeds),
    }

    out_path = config.LOGS_DIR / "robustness_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved robustness summary to {out_path}")

