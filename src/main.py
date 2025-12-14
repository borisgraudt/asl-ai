import argparse
import sys
import cv2
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision import HandTracker, LandmarkPreprocessor
from src.ai import GestureClassifier
from src.ui import ASLVisualizer, Camera
from src.utils.config import config
from src.utils.logger import setup_logger, get_logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="ASL&AI - Real-time American Sign Language Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    # Run with visualization
  python src/main.py --no-visual        # Run without visualization
  python src/main.py --camera 1         # Use camera index 1
  python src/main.py --benchmark        # Run in benchmark mode
        """
    )
    
    parser.add_argument(
        "--visual",
        action="store_true",
        default=True,
        help="Enable visualization window (default: True)"
    )
    
    parser.add_argument(
        "--no-visual",
        dest="visual",
        action="store_false",
        help="Disable visualization window"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help=f"Camera index (default: {config.CAMERA_CONFIG['camera_index']})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to model file (default: {config.MODEL_PATH})"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=f"Confidence threshold (default: {config.INFERENCE_CONFIG['confidence_threshold']})"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (no visualization, print metrics)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def validate_setup() -> bool:
    """
    Validate that required files and dependencies are available.
    
    Returns
    -------
    bool
        True if setup is valid, False otherwise
    """
    logger = get_logger(__name__)
    
    # Check model files
    model_paths = config.get_model_paths()
    missing_files = []
    
    for name, path in model_paths.items():
        if name == "tflite":  # TFLite is optional
            continue
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        logger.error("Missing required model files:")
        for file_info in missing_files:
            logger.error(f"  - {file_info}")
        logger.error("\nPlease train the model first:")
        logger.error("  make train")
        logger.error("  or")
        logger.error("  python scripts/train.py")
        return False
    
    logger.info("Setup validation passed")
    return True


def run_recognition(
    visual: bool = True,
    camera_index: int = None,
    model_path: str = None,
    confidence_threshold: float = None,
    benchmark: bool = False
) -> None:
    """
    Run real-time ASL recognition.
    
    Parameters
    ----------
    visual : bool
        Whether to show visualization window
    camera_index : int, optional
        Camera device index
    model_path : str, optional
        Path to model file
    confidence_threshold : float, optional
        Confidence threshold for predictions
    benchmark : bool
        Run in benchmark mode (no visualization)
    """
    logger = get_logger(__name__)
    
    # Override config with CLI arguments
    if camera_index is not None:
        config.CAMERA_CONFIG["camera_index"] = camera_index
    
    if confidence_threshold is not None:
        config.INFERENCE_CONFIG["confidence_threshold"] = confidence_threshold
    
    # Initialize components
    logger.info("Initializing ASL recognition system...")
    
    try:
        # Initialize classifier
        classifier = GestureClassifier(
            model_path=Path(model_path) if model_path else None,
            confidence_threshold=config.INFERENCE_CONFIG["confidence_threshold"],
            history_length=config.INFERENCE_CONFIG["prediction_history_length"]
        )
        logger.info("Gesture classifier initialized")
        
        # Initialize hand tracker
        tracker = HandTracker()
        logger.info("Hand tracker initialized")
        
        # Initialize preprocessor
        preprocessor = LandmarkPreprocessor(scaler=classifier.scaler)
        logger.info("Preprocessor initialized")
        
        # Initialize camera
        camera = Camera(camera_index=config.CAMERA_CONFIG["camera_index"])
        if not camera.open():
            logger.error("Failed to open camera. Exiting.")
            return
        
        # Initialize visualizer (if enabled)
        visualizer = None
        if visual and not benchmark:
            visualizer = ASLVisualizer(show_fps=True)
            cv2.namedWindow(config.VISUALIZATION_CONFIG["window_name"], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                config.VISUALIZATION_CONFIG["window_name"],
                config.CAMERA_CONFIG["frame_width"],
                config.CAMERA_CONFIG["frame_height"] + config.VISUALIZATION_CONFIG["info_panel_height"]
            )
            logger.info("Visualization enabled")
        
        logger.info("System ready. Press 'q' to quit.")
        
        # Main loop
        frame_count = 0
        while camera.is_opened():
            # Read frame
            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame
            landmarks = tracker.get_landmarks(frame)
            
            if landmarks is not None:
                # Draw landmarks on frame
                if visualizer:
                    tracker.draw_landmarks(frame, landmarks)
                
                # Preprocess landmarks
                features = preprocessor.process(landmarks)
                
                # Classify gesture
                letter, confidence, stability = classifier.predict(features)
                
                # Log prediction (in benchmark mode or if confidence is high)
                if benchmark or confidence >= config.INFERENCE_CONFIG["confidence_threshold"]:
                    logger.debug(f"Prediction: {letter} (confidence: {confidence:.3f}, stability: {stability:.3f})")
                
                # Update visualization
                if visualizer:
                    frame = visualizer.draw_overlay(frame, letter, confidence, stability)
            else:
                # No hand detected
                if visualizer:
                    frame = visualizer.show_no_hand_message(frame)
            
            # Display frame
            if visualizer:
                cv2.imshow(config.VISUALIZATION_CONFIG["window_name"], frame)
            
            # Handle keyboard input
            if visualizer:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
            elif benchmark and frame_count >= 1000:  # Run for 1000 frames in benchmark mode
                logger.info("Benchmark complete (1000 frames)")
                break
        
        # Cleanup
        logger.info("Shutting down...")
        camera.release()
        if visualizer:
            cv2.destroyAllWindows()
        tracker.release()
        
        # Print final metrics
        if visualizer:
            metrics = visualizer.performance_monitor.get_metrics()
            logger.info(f"Final metrics - FPS: {metrics['fps']:.2f}, Latency: {metrics['latency_ms']:.2f}ms")
        
        logger.info("Shutdown complete")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during recognition: {e}", exc_info=True)
        raise


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(__import__("logging"), args.log_level)
    setup_logger("asl_ai", log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("ASL&AI - Real-time American Sign Language Recognition")
    logger.info("=" * 60)
    
    # Ensure directories exist
    config.ensure_directories()
    
    # Validate setup
    if not validate_setup():
        sys.exit(1)
    
    # Run recognition
    run_recognition(
        visual=args.visual and not args.benchmark,
        camera_index=args.camera,
        model_path=args.model,
        confidence_threshold=args.confidence,
        benchmark=args.benchmark
    )


if __name__ == "__main__":
    main()

