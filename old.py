import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import threading
from queue import Queue
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
from functools import wraps



def deprecated(message):
    """Decorator to mark functions as deprecated"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(f"Deprecated: {message}", DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class ObjectDetector:
    def __init__(self):
        # Camera settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Initialize MediaPipe solutions with performance optimizations
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face detection optimized
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # Use faster model
            min_detection_confidence=0.7
        )
        
        # Hand detection optimized
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,  # Use fastest model
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2  # Limit to 2 hands for performance
        )
        
        # Pose detection optimized
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Use fastest model
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        

        # Processing optimization
        self.processing_size = (640, 360)  # Smaller processing size
        self.display_size = (1280, 720)    # Original display size
        
        # Colors for visualization
        self.colors = {
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
        
        # Colors for hands (separate for clarity)
        self.hand_colors = {
            'Left': (255, 0, 0),    # Blue for left hand
            'Right': (0, 255, 0)    # Green for right hand
        }
        
        # Tracking history for smoothing
        self.prev_detections = {
            'faces': [],
            'hands': [],
            'poses': []
        }
        self.smoothing_factor = 0.7
        
        # Processing queues and stats
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.frame_timestamps = {}
        self.processing_stats = {
            'processed_frames': 0,
            'dropped_frames': 0,
            'avg_latency': 0
        }
        
        # Initialize thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        # Mark drawing-related attributes as deprecated
        warnings.warn(
            "Drawing feature is deprecated and will be removed in a future version.",
            DeprecationWarning
        )

    def _smooth_detections(self, current, previous, detection_type):
        """Smooth detections to reduce jitter"""
        if not previous or not current:
            return current
        
        smoothed = []
        for curr in current:
            # Find matching previous detection
            prev = None
            if detection_type == 'hands':
                prev = next((p for p in previous if p['id'] == curr['id']), None)
            elif detection_type == 'faces':
                # Match faces by closest centers
                curr_center = (curr['bbox'][0] + curr['bbox'][2]//2, 
                             curr['bbox'][1] + curr['bbox'][3]//2)
                min_dist = float('inf')
                for p in previous:
                    p_center = (p['bbox'][0] + p['bbox'][2]//2, 
                              p['bbox'][1] + p['bbox'][3]//2)
                    dist = ((curr_center[0] - p_center[0])**2 + 
                           (curr_center[1] - p_center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        prev = p
            
            if prev:
                # Smooth landmarks
                if 'landmarks' in curr and 'landmarks' in prev:
                    curr_landmarks = np.array(curr['landmarks'])
                    prev_landmarks = np.array(prev['landmarks'])
                    smoothed_landmarks = (prev_landmarks * self.smoothing_factor + 
                                       curr_landmarks * (1 - self.smoothing_factor))
                    curr['landmarks'] = smoothed_landmarks.astype(int).tolist()
                
                # Smooth bounding box
                if 'bbox' in curr and 'bbox' in prev:
                    curr_bbox = np.array(curr['bbox'])
                    prev_bbox = np.array(prev['bbox'])
                    smoothed_bbox = (prev_bbox * self.smoothing_factor + 
                                   curr_bbox * (1 - self.smoothing_factor))
                    curr['bbox'] = tuple(smoothed_bbox.astype(int))
            
            smoothed.append(curr)
        
        return smoothed

    def _process_frame(self, frame):
        """Optimized frame processing"""
        # Resize for processing
        frame_small = cv2.resize(frame, self.processing_size)
        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        
        # Process with all detectors
        faces = self._detect_faces(rgb_frame)
        hands = self._detect_hands(rgb_frame)
        pose = self._detect_pose(rgb_frame)
        
        # Scale detections back to display size
        scale_x = self.display_size[0] / self.processing_size[0]
        scale_y = self.display_size[1] / self.processing_size[1]
        
        faces = self._scale_detections(faces, scale_x, scale_y)
        hands = self._scale_detections(hands, scale_x, scale_y)
        if pose:
            pose = self._scale_detections([pose], scale_x, scale_y)[0]
        
        return frame, faces, hands, pose

    def _scale_detections(self, detections, scale_x, scale_y):
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

    def _detect_faces(self, rgb_frame):
        """Enhanced face detection for multiple faces"""
        results = self.face_detector.process(rgb_frame)
        faces = []
        
        if results.detections:
            frame_height, frame_width = rgb_frame.shape[:2]
            for idx, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                face_data = {
                    'id': idx,
                    'bbox': (
                        int(bbox.xmin * frame_width),
                        int(bbox.ymin * frame_height),
                        int(bbox.width * frame_width),
                        int(bbox.height * frame_height)
                    ),
                    'confidence': detection.score[0],
                    'landmarks': [
                        (int(landmark.x * frame_width), int(landmark.y * frame_height))
                        for landmark in detection.location_data.relative_keypoints
                    ]
                }
                faces.append(face_data)
        
        return faces

    def _detect_pointing(self, landmarks):
        """Detect if hand is pointing based on finger positions"""
        # Index finger points
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        
        # Middle finger points
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Calculate if index is extended and middle is curled
        index_extended = index_tip[1] < index_pip[1]  # Y coordinate comparison
        middle_curled = middle_tip[1] > middle_pip[1]
        
        return index_extended and middle_curled

    @deprecated("Drawing feature will be removed in future versions")
    def _is_point_different(self, p1, p2, min_distance=5):
        """Check if two points are significantly different to avoid duplicate points"""
        return abs(p1[0] - p2[0]) > min_distance or abs(p1[1] - p2[1]) > min_distance

    def _detect_hands(self, rgb_frame):
        """Enhanced hand detection"""
        results = self.hand_detector.process(rgb_frame)
        hands = []
        
        if results.multi_hand_landmarks:
            frame_height, frame_width = rgb_frame.shape[:2]
            
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Get landmarks with precise coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    landmarks.append((x, y))
                
                # Detect pointing gesture
                is_pointing = self._detect_pointing(landmarks)
                
                hand_data = {
                    'landmarks': landmarks,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score,
                    'connections': self.mp_hands.HAND_CONNECTIONS,
                    'id': idx,
                    'is_pointing': is_pointing,
                    'index_tip': landmarks[8]  # Store index fingertip position
                }
                
                # Calculate bounding box
                x_coords = [l[0] for l in landmarks]
                y_coords = [l[1] for l in landmarks]
                hand_data['bbox'] = (
                    min(x_coords),
                    min(y_coords),
                    max(x_coords) - min(x_coords),
                    max(y_coords) - min(y_coords)
                )
                
                hands.append(hand_data)
        
        return hands

    def _detect_pose(self, rgb_frame):
        """Body pose detection using MediaPipe"""
        results = self.pose_detector.process(rgb_frame)
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

    def _draw_results(self, frame, faces, hands, pose):
        """Enhanced visualization"""
        output = frame.copy()
        
        # Draw hands with visualization of pointing
        for hand in hands:
            color = self.hand_colors[hand['handedness']]
            
            # Draw hand skeleton
            for connection in hand['connections']:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = hand['landmarks'][start_idx]
                end_point = hand['landmarks'][end_idx]
                cv2.line(output, start_point, end_point, color, 2, cv2.LINE_AA)
            
            # Highlight index fingertip when pointing
            if hand['is_pointing']:
                cv2.circle(output, hand['index_tip'], 8, (0, 0, 255), -1, cv2.LINE_AA)
            
            # Add pointing indicator to label
            x, y, w, h = hand['bbox']
            label = f"{hand['handedness']}"
            if hand['is_pointing']:
                label += " (Pointing)"
            cv2.putText(output, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Draw faces
        for face in faces:
            color = self.colors['faces'][face['id'] % len(self.colors['faces'])]
            x, y, w, h = face['bbox']
            
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2, cv2.LINE_AA)
            cv2.putText(output, f"Face {face['id']}: {face['confidence']:.2f}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            
            for landmark in face['landmarks']:
                cv2.circle(output, landmark, 2, color, -1, cv2.LINE_AA)
        
        # Draw pose
        if pose:
            for connection in pose['connections']:
                start_idx = connection[0]
                end_idx = connection[1]
                if pose['visibility'][start_idx] > 0.65 and pose['visibility'][end_idx] > 0.65:
                    color = self.colors['pose'][0]
                    cv2.line(output, pose['landmarks'][start_idx],
                            pose['landmarks'][end_idx], color, 2, cv2.LINE_AA)
            
            for i, landmark in enumerate(pose['landmarks']):
                if pose['visibility'][i] > 0.65:
                    cv2.circle(output, landmark, 3, (0, 0, 255), -1, cv2.LINE_AA)
        
        # Draw stats
        stats = [
            f"Faces: {len(faces)}",
            f"Hands: {len(hands)}",
            f"FPS: {1.0/self.processing_stats['avg_latency']:.0f}"
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(output, text, (10, 25 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return output

    def _processing_loop(self):
        """Background processing loop"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                frame_time = self.frame_timestamps.get(id(frame))
                
                processed_frame, faces, hands, pose = self._process_frame(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put((processed_frame, faces, hands, pose, frame_time))
                    self.processing_stats['processed_frames'] += 1
                else:
                    self.processing_stats['dropped_frames'] += 1
                
                self.frame_queue.task_done()
                
            except Exception as e:
                if self.running:
                    print(f"Processing error: {str(e)}")
                time.sleep(0.001)

    def _main_loop(self):
        """Main loop"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                self.frame_timestamps[id(frame)] = time.time()
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    self.processing_stats['dropped_frames'] += 1
                    continue

                if not self.result_queue.empty():
                    processed_frame, faces, hands, pose, frame_time = self.result_queue.get()
                    
                    current_latency = time.time() - frame_time
                    self.processing_stats['avg_latency'] = (
                        0.9 * self.processing_stats['avg_latency'] + 
                        0.1 * current_latency
                    )
                    
                    output_frame = self._draw_results(processed_frame, faces, hands, pose)
                    self.result_queue.task_done()
                    
                    cv2.imshow('MediaPipe Detection - Press Q to quit', output_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"Main loop error: {str(e)}")
                time.sleep(0.001)

    def start(self):
        """Start the detection system"""
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.processing_thread.start()
        self._main_loop()
        self.stop()

    def stop(self):
        """Clean up resources"""
        self.running = False
        
        # Clean up queues
        while not self.frame_queue.empty():
            self.frame_queue.get()
            self.frame_queue.task_done()
            
        while not self.result_queue.empty():
            self.result_queue.get()
            self.result_queue.task_done()
        
        # Close detectors
        self.face_detector.close()
        self.hand_detector.close()
        self.pose_detector.close()
        
        self.processing_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector()
    try:
        detector.start()
    finally:
        detector.stop()