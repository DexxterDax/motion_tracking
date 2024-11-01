import cv2
import time
import numpy as np
from typing import Optional, Tuple

class Camera:
    def __init__(self, camera_id: int = 0, width: int = 1920, height: int = 1080):
        """Initialize camera with specified resolution."""
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.dropped_frames = 0

    def start(self) -> bool:
        """Start the camera capture."""
        try:
            # Try to open camera with default backend
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                # If default fails, try different backends
                backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_DSHOW]
                for backend in backends:
                    self.cap = cv2.VideoCapture(self.camera_id + backend)
                    if self.cap.isOpened():
                        break

            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm up camera
            for _ in range(5):
                self.cap.read()

            self.last_frame_time = time.time()
            return True

        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        try:
            if not self.cap or not self.cap.isOpened():
                return False, None

            ret, frame = self.cap.read()
            current_time = time.time()

            if ret:
                self.frame_count += 1
                self.last_frame_time = current_time
                return True, frame
            else:
                self.dropped_frames += 1
                return False, None

        except Exception as e:
            print(f"Camera read error: {str(e)}")
            self.dropped_frames += 1
            return False, None

    def get_stats(self) -> dict:
        """Get camera performance statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        fps = self.frame_count / (elapsed + 1e-6)
        
        return {
            'fps': fps,
            'latency': elapsed,
            'dropped': self.dropped_frames
        }

    def release(self):
        """Release the camera resources."""
        try:
            if self.cap:
                self.cap.release()
            self.cap = None
        except Exception as e:
            print(f"Camera release error: {str(e)}") 