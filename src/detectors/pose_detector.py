import cv2
import mediapipe as mp
import numpy as np
from .base_detector import BaseDetector
from detector_types import PoseDetection
from typing import Optional

class PoseDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def process(self, frame: np.ndarray) -> Optional[PoseDetection]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        pose_data = None
        
        if results.pose_landmarks:
            frame_height, frame_width = rgb_frame.shape[:2]
            pose_data = {
                'landmarks': [
                    (int(landmark.x * frame_width), int(landmark.y * frame_height))
                    for landmark in results.pose_landmarks.landmark
                ],
                'connections': self.mp_pose.POSE_CONNECTIONS,
                'visibility': [landmark.visibility for landmark in results.pose_landmarks.landmark]
            }
        
        return pose_data

    def close(self):
        self.detector.close()