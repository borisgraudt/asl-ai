"""
MediaPipe hand tracking wrapper.

Provides a clean interface for hand detection and landmark extraction
using Google's MediaPipe framework.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HandTracker:
    """
    MediaPipe hand tracking wrapper for real-time hand detection.
    
    This class provides a clean interface for detecting hands and extracting
    landmarks from video frames using Google's MediaPipe Hands solution.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize MediaPipe hand tracker.
        
        Parameters
        ----------
        **kwargs
            Override default MediaPipe configuration:
            - static_image_mode: bool (default: False)
            - max_num_hands: int (default: 1)
            - min_detection_confidence: float (default: 0.7)
            - min_tracking_confidence: float (default: 0.7)
        """
        # Merge user config with defaults
        mp_config = {**config.MEDIAPIPE_CONFIG, **kwargs}
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(**mp_config)
        
        logger.info(f"Initialized HandTracker with config: {mp_config}")
    
    def process_frame(self, frame: np.ndarray) -> Optional[Any]:
        """
        Process a video frame to detect hands.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format (OpenCV default)
        
        Returns
        -------
        Optional[Any]
            MediaPipe results object if hands detected, None otherwise
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        return results
    
    def get_landmarks(self, frame: np.ndarray) -> Optional[Any]:
        """
        Extract hand landmarks from frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame in BGR format
        
        Returns
        -------
        Optional[Any]
            First hand landmarks if detected, None otherwise
        """
        results = self.process_frame(frame)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Any,
        connections: bool = True,
        style: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Draw hand landmarks and connections on frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on (modified in-place)
        landmarks : Any
            MediaPipe hand landmarks
        connections : bool
            Whether to draw connections between landmarks
        style : Optional[Dict[str, Any]]
            Custom drawing style. If None, uses default colors from config.
        
        Returns
    -------
        np.ndarray
            Frame with landmarks drawn
        """
        if style is None:
            vis_config = config.VISUALIZATION_CONFIG
            connection_spec = mp.solutions.drawing_utils.DrawingSpec(
                color=vis_config["landmark_color"],
                thickness=3,
                circle_radius=1
            )
            landmark_spec = mp.solutions.drawing_utils.DrawingSpec(
                color=vis_config["connection_color"],
                thickness=5,
                circle_radius=5
            )
        else:
            connection_spec = style.get("connection", None)
            landmark_spec = style.get("landmark", None)
        
        if connections:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                connection_spec,
                landmark_spec
            )
        else:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                None,  # No connections
                None,
                landmark_spec
            )
        
        return frame
    
    def release(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
        logger.debug("HandTracker resources released")


