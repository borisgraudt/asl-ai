"""
UI and visualization module for ASL recognition application.

Provides real-time visualization with FPS tracking, confidence display,
and performance metrics overlay.
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Any, Tuple
from collections import deque

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring for FPS and latency tracking.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Parameters
        ----------
        window_size : int
            Number of frames to average for FPS calculation
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.fps = 0.0
        self.latency = 0.0
    
    def update(self) -> None:
        """Update FPS and latency metrics."""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.last_time = current_time
        
        if len(self.frame_times) > 0:
            self.fps = 1.0 / np.mean(self.frame_times)
            self.latency = np.mean(self.frame_times) * 1000  # Convert to ms
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with 'fps' and 'latency_ms' keys
        """
        return {
            "fps": self.fps,
            "latency_ms": self.latency,
        }


class ASLVisualizer:
    """
    Real-time visualization for ASL recognition.
    
    Provides OpenCV-based visualization with hand landmarks,
    predictions, confidence scores, and performance metrics.
    """
    
    def __init__(self, show_fps: bool = True):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        show_fps : bool
            Whether to display FPS counter
        """
        self.show_fps = show_fps
        self.performance_monitor = PerformanceMonitor()
        self.vis_config = config.VISUALIZATION_CONFIG
        
        logger.info("ASLVisualizer initialized")
    
    def create_info_panel(
        self,
        width: int,
        letter: Optional[str] = None,
        confidence: Optional[float] = None,
        stability: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create information panel overlay.
        
        Parameters
        ----------
        width : int
            Width of the panel (should match frame width)
        letter : Optional[str]
            Predicted letter
        confidence : Optional[float]
            Prediction confidence [0, 1]
        stability : Optional[float]
            Prediction stability [0, 1]
        
        Returns
        -------
        np.ndarray
            Information panel image
        """
        height = self.vis_config["info_panel_height"]
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = self.vis_config["text_color"]
        
        if letter:
            # Display predicted letter
            cv2.putText(
                panel,
                f"Letter: {letter}",
                (20, y_offset),
                font,
                2,
                color,
                3
            )
            y_offset += 60
        
        if confidence is not None:
            # Display confidence
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(
                panel,
                conf_text,
                (20, y_offset),
                font,
                1,
                color,
                2
            )
            y_offset += 50
        
        if stability is not None:
            # Display stability
            stab_text = f"Stability: {stability*100:.1f}%"
            cv2.putText(
                panel,
                stab_text,
                (20, y_offset),
                font,
                1,
                color,
                2
            )
            y_offset += 50
        
        # Display FPS if enabled
        if self.show_fps:
            metrics = self.performance_monitor.get_metrics()
            fps_text = f"FPS: {metrics['fps']:.1f} | Latency: {metrics['latency_ms']:.1f}ms"
            cv2.putText(
                panel,
                fps_text,
                (20, y_offset),
                font,
                0.8,
                color,
                2
            )
        
        return panel
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        letter: Optional[str] = None,
        confidence: Optional[float] = None,
        stability: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw complete overlay on frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
        letter : Optional[str]
            Predicted letter
        confidence : Optional[float]
            Prediction confidence
        stability : Optional[float]
            Prediction stability
        
        Returns
        -------
        np.ndarray
            Frame with overlay
        """
        # Update performance metrics
        self.performance_monitor.update()
        
        # Create info panel
        info_panel = self.create_info_panel(
            frame.shape[1],
            letter,
            confidence,
            stability
        )
        
        # Combine frame and panel
        combined = np.vstack([frame, info_panel])
        
        return combined
    
    def show_no_hand_message(self, frame: np.ndarray) -> np.ndarray:
        """
        Display message when no hand is detected.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
        
        Returns
        -------
        np.ndarray
            Frame with message
        """
        info_panel = self.create_info_panel(frame.shape[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = self.vis_config["text_color"]
        
        cv2.putText(
            info_panel,
            "Show the gesture to the camera",
            (20, 50),
            font,
            1,
            color,
            2
        )
        
        combined = np.vstack([frame, info_panel])
        return combined


class Camera:
    """
    Camera interface wrapper for OpenCV VideoCapture.
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera.
        
        Parameters
        ----------
        camera_index : int
            Camera device index
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.cam_config = config.CAMERA_CONFIG
        
        logger.info(f"Initializing camera {camera_index}")
    
    def open(self) -> bool:
        """
        Open camera connection.
        
        Returns
        -------
        bool
            True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_config["frame_height"])
            self.cap.set(cv2.CAP_PROP_FPS, self.cam_config["fps_target"])
            
            logger.info("Camera opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera.
        
        Returns
        -------
        tuple[bool, Optional[np.ndarray]]
            (success, frame) tuple
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame from camera")
            return False, None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        return True, frame
    
    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.cap is not None and self.cap.isOpened()

