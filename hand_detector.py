# import cv2
# import mediapipe as mp
# import numpy as np
# from .src.detectors.base_detector import BaseDetector
# from detector_types import HandDetection
# from typing import Optional, Tuple, List, Dict
# import time

# class HandDetector(BaseDetector):
#     def __init__(self):
#         super().__init__()
#         self.mp_hands = mp.solutions.hands
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.mp_drawing_styles = mp.solutions.drawing_styles
        
#         # Store hand connections for visualization
#         self._connections = self.mp_hands.HAND_CONNECTIONS
        
#         # Initialize hand detector with optimized parameters
#         self.detector = self.mp_hands.Hands(
#             static_image_mode=False,
#             model_complexity=1,      # Balance between speed and accuracy
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5,
#             max_num_hands=2
#         )
        
#         # # Gesture detection parameters
#         # self.PINCH_THRESHOLD = 0.03
#         # self.FINGER_EXTENSION_THRESHOLD = 0.1
        
#         # # Tracking and smoothing
#         # self._tracking_history = []
#         # self._max_history = 5
#         # self._smoothing_factor = 0.7
        
#         # Performance optimization
#         self._last_detection_time = 0
#         self._min_detection_interval = 1/30  # Max 30 FPS processing

#     @property
#     def connections(self):
#         """Returns the hand connections for visualization."""
#         return self._connections

#     def process(self, frame: np.ndarray) -> List[HandDetection]:
#         """Process a frame to detect hands."""
#         # Return empty list if no frame
#         if frame is None:
#             return []
            
#         current_time = time.time()
#         if current_time - self._last_detection_time < self._min_detection_interval:
#             return []

#         try:
#             # Preprocess frame
#             if frame.shape[0] > 480:  # Resize large frames
#                 scale = 480 / frame.shape[0]
#                 frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
#             # Convert to RGB
#             frame = cv2.flip(frame, 1)  # Mirror image
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Detect hands
#             results = self.detector.process(rgb_frame)
#             hands = []
            
#             if results and results.multi_hand_landmarks:
#                 frame_dims = np.array([frame.shape[1], frame.shape[0]])
                
#                 for idx, (hand_landmarks, handedness) in enumerate(
#                     zip(results.multi_hand_landmarks, results.multi_handedness)
#                 ):
#                     # Extract landmarks
#                     landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
#                     landmarks_2d = np.column_stack([
#                         (1 - landmarks[:, 0]) * frame_dims[0],  # Mirrored x
#                         landmarks[:, 1] * frame_dims[1]
#                     ]).astype(int)
                    
#                     # Process hand data
#                     hand_data = self._process_hand_data(
#                         landmarks, landmarks_2d, handedness, idx, frame_dims
#                     )
#                     if hand_data:  # Only append if we got valid data
#                         hands.append(hand_data)
            
#             self._last_detection_time = current_time
#             return hands

#         except Exception as e:
#             print(f"Processing error: {str(e)}")
#             return []  # Return empty list on error

#     def _process_hand_data(self, landmarks_3d, landmarks_2d, handedness, idx, frame_dims):
#         """Process detected hand data into a structured format."""
#         try:
#             # Convert landmarks to integer tuples for OpenCV compatibility
#             landmarks_2d_tuples = [(int(x), int(y)) for x, y in landmarks_2d]
            
#             # Get basic hand information
#             hand_info = {
#                 'landmarks': landmarks_2d_tuples,
#                 'landmarks_3d': landmarks_3d.tolist(),
#                 'handedness': "Left" if handedness.classification[0].label == "Right" else "Right",
#                 'confidence': float(handedness.classification[0].score),
#                 'id': idx,
#                 'connections': self.mp_hands.HAND_CONNECTIONS,
#                 'index_tip': landmarks_2d_tuples[8]  # Add index fingertip position
#             }
            
#             # Calculate bounding box
#             x_coords = [x for x, y in landmarks_2d_tuples]
#             y_coords = [y for x, y in landmarks_2d_tuples]
#             hand_info['bbox'] = (
#                 min(x_coords),
#                 min(y_coords),
#                 max(x_coords) - min(x_coords),
#                 max(y_coords) - min(y_coords)
#             )
            
#             return hand_info

#         except Exception as e:
#             print(f"Hand data processing error: {str(e)}")
#             return None

#     def _detect_gestures(self, landmarks_3d):
#         """Detect hand gestures."""
#         # Detect pinch
#         thumb_tip = landmarks_3d[4]
#         index_tip = landmarks_3d[8]
#         pinch_distance = np.linalg.norm(thumb_tip - index_tip)
#         pinch = pinch_distance < self.PINCH_THRESHOLD

#         # Get finger states
#         finger_states = self._get_finger_states(landmarks_3d)
        
#         # Detect gestures based on finger states
#         fist = not np.any(finger_states)
#         open_palm = np.all(finger_states)
#         pointing = np.array_equal(finger_states, [False, True, False, False, False])

#         # Return all gesture states directly in the main dict
#         return {
#             'pinch': pinch,
#             'fist': fist,
#             'open_palm': open_palm,
#             'pointing': pointing,
#             'is_pointing': pointing,  # Add is_pointing flag for compatibility
#             'gestures': {  # Also include nested gestures dict for compatibility
#                 'pinch': pinch,
#                 'fist': fist,
#                 'open_palm': open_palm,
#                 'pointing': pointing
#             }
#         }

#     def _get_finger_states(self, landmarks_3d):
#         """Get the state of each finger (extended or not)."""
#         finger_tips = [4, 8, 12, 16, 20]
#         finger_pips = [3, 7, 11, 15, 19]
        
#         states = []
#         for tip, pip in zip(finger_tips, finger_pips):
#             # Check if finger is extended
#             extended = landmarks_3d[tip][1] < landmarks_3d[pip][1]
#             states.append(extended)
            
#         # Special case for thumb
#         thumb_angle = self._calculate_thumb_angle(landmarks_3d)
#         states[0] = thumb_angle > 2.1
        
#         return states

#     def _calculate_thumb_angle(self, landmarks_3d):
#         """Calculate the angle of the thumb."""
#         cmc = landmarks_3d[1]
#         mcp = landmarks_3d[2]
#         tip = landmarks_3d[4]
        
#         v1 = mcp - cmc
#         v2 = tip - mcp
        
#         cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#         angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
#         return angle

#     def _calculate_orientation(self, landmarks_3d):
#         """Calculate the orientation of the hand."""
#         wrist = landmarks_3d[0]
#         middle_mcp = landmarks_3d[9]
        
#         direction = middle_mcp - wrist
        
#         x_angle = np.arctan2(direction[1], direction[0])
#         y_angle = np.arctan2(direction[2], direction[0])
#         z_angle = np.arctan2(direction[2], direction[1])
        
#         return {
#             'x': float(x_angle),
#             'y': float(y_angle),
#             'z': float(z_angle)
#         }

#     def _update_tracking_history(self, hands):
#         """Update tracking history for smoothing."""
#         self._tracking_history.append(hands)
#         if len(self._tracking_history) > self._max_history:
#             self._tracking_history.pop(0)

#     def _smooth_predictions(self):
#         """Apply temporal smoothing to predictions."""
#         if not self._tracking_history:
#             return []
        
#         if len(self._tracking_history) < 2:
#             return self._tracking_history[-1]
        
#         smoothed_hands = []
#         latest_hands = self._tracking_history[-1]
        
#         for hand in latest_hands:
#             hand_id = hand['id']
#             previous_matches = [
#                 prev_hand for prev_frame in self._tracking_history[:-1]
#                 for prev_hand in prev_frame
#                 if prev_hand['id'] == hand_id
#             ]
            
#             if not previous_matches:
#                 smoothed_hands.append(hand)
#                 continue
            
#             smoothed_hand = hand.copy()
            
#             # Smooth landmarks while maintaining tuple format
#             if 'landmarks' in hand:
#                 current = np.array(hand['landmarks'])
#                 prev = np.array(previous_matches[-1]['landmarks'])
#                 smoothed = (
#                     self._smoothing_factor * current +
#                     (1 - self._smoothing_factor) * prev
#                 ).astype(int)
#                 smoothed_hand['landmarks'] = [tuple(point) for point in smoothed]
            
#             # Smooth 3D landmarks
#             if 'landmarks_3d' in hand:
#                 current = np.array(hand['landmarks_3d'])
#                 prev = np.array(previous_matches[-1]['landmarks_3d'])
#                 smoothed = (
#                     self._smoothing_factor * current +
#                     (1 - self._smoothing_factor) * prev
#                 )
#                 smoothed_hand['landmarks_3d'] = smoothed.tolist()
            
#             smoothed_hands.append(smoothed_hand)
        
#         return smoothed_hands

#     def close(self):
#         """Release resources."""
#         self.detector.close()