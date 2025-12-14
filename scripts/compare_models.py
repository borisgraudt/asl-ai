import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.train import train_model
from src.utils.config import config
from src.utils.logger import setup_logger, get_logger


def _run(arch: str, overrides: dict) -> dict:
    # Apply overrides via environment variables (config reads env on import time for some keys,
    # but TRAINING_CONFIG is already loaded; we update it here explicitly for this run).
    config.TRAINING_CONFIG.update(overrides)
    run_name = f"compare_{arch}"
    model_path = config.MODELS_DIR / f"{run_name}.h5"
    return train_model(model_path=model_path, save_plots=True, run_name=run_name)


if __name__ == "__main__":
    setup_logger("asl_ai.compare", log_level=20)
    logger = get_logger(__name__)

    logger.info("Comparing architectures: MLP vs MoE")

    results = {}

    # Baseline
    results["mlp"] = _run(
        "mlp",
        {
            "architecture": "mlp",
        },
    )

    # MoE
    results["moe"] = _run(
        "moe",
        {
            "architecture": "moe",
            "moe_num_experts": int(os.getenv("ASL_AI_MOE_EXPERTS", "6")),
            "moe_expert_units": int(os.getenv("ASL_AI_MOE_UNITS", "128")),
            "moe_top_k": int(os.getenv("ASL_AI_MOE_TOPK", "2")),
        },
    )

    # Write summary
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LOGS_DIR / "compare_models.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved comparison results to {out_path}")
    logger.info(f"MLP accuracy: {results['mlp']['test_accuracy']:.4f} | top3: {results['mlp']['test_top3_accuracy']:.4f}")
    logger.info(f"MoE accuracy: {results['moe']['test_accuracy']:.4f} | top3: {results['moe']['test_top3_accuracy']:.4f}")


