import sys
import json
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.train import plot_training_history
from src.utils.config import config
from src.utils.logger import setup_logger, get_logger

def load_history_from_json(json_path: Path) -> dict:
    # Load training history from JSON file.
    with open(json_path, 'r') as f:
        return json.load(f)

def load_history_from_pickle(pickle_path: Path) -> dict:
    # Load training history from pickle file.
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    setup_logger("asl_ai.regenerate_plot", log_level=20)
    logger = get_logger(__name__)
    
    logger.info("Regenerating training history plot with English labels...")
    
    # Check for saved history files
    history_json = project_root / "logs" / "training_history.json"
    history_pkl = project_root / "logs" / "training_history.pkl"
    
    history = None
    
    if history_json.exists():
        logger.info(f"Loading history from {history_json}")
        history = load_history_from_json(history_json)
    elif history_pkl.exists():
        logger.info(f"Loading history from {history_pkl}")
        history = load_history_from_pickle(history_pkl)
    else:
        logger.warning("No saved training history found.")
        logger.info("To regenerate the plot, you need to:")
        logger.info("1. Retrain the model (the plot will be generated automatically)")
        logger.info("2. Or save the training history during training")
        logger.info("")
        logger.info("Example: After training, save history.history to a JSON or pickle file")
        sys.exit(1)
    
    # Regenerate plot
    plot_path = config.PLOTS_DIR / "training_history.png"
    plot_training_history(history, save_path=plot_path)
    logger.info(f"Plot regenerated successfully: {plot_path}")

