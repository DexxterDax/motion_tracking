import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseDetector(ABC):
    def __init__(self):
        self.processing_size: Tuple[int, int] = (640, 360)
        self.display_size: Tuple[int, int] = (1280, 720)
        self.colors: Dict[str, Any] = {
            'faces': [
                (255, 0, 0),    # Blue
                (255, 128, 0),  # Light Blue
                (255, 200, 0)   # Very Light Blue
            ],
            'pose': [
                (0, 255, 0),    # Green
                (255, 128, 0),  # Orange
                (0, 128, 255)   # Light Blue
            ]
        }
        self.hand_colors = {
            'Left': (255, 0, 0),    # Blue for left hand
            'Right': (0, 255, 0)    # Green for right hand
        }
        self.smoothing_factor = 0.7

    def _scale_detections(self, detections: list, scale_x: float, scale_y: float) -> list:
        """Scale detection coordinates"""
        if not detections:
            return detections
        
        scaled = []
        for det in detections:
            scaled_det = det.copy()
            
            if 'landmarks' in det:
                scaled_det['landmarks'] = [
                    (int(x * scale_x), int(y * scale_y))
                    for x, y in det['landmarks']
                ]
            
            if 'bbox' in det:
                x, y, w, h = det['bbox']
                scaled_det['bbox'] = (
                    int(x * scale_x),
                    int(y * scale_y),
                    int(w * scale_x),
                    int(h * scale_y)
                )
            
            scaled.append(scaled_det)
        
        return scaled

    @abstractmethod
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame and return detections"""
        pass