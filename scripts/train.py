import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.train import train_model
from src.utils.logger import setup_logger, get_logger

if __name__ == "__main__":
    # Setup logging
    setup_logger("asl_ai.train", log_level=20)  # INFO level
    logger = get_logger(__name__)
    
    logger.info("Starting model training...")
    logger.info("=" * 60)
    
    try:
        results = train_model()
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Test Accuracy: {results['test_accuracy']*100:.2f}%")
        logger.info(f"Training Time: {results['training_time_minutes']:.2f} minutes")
        logger.info(f"Epochs: {results['num_epochs']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
