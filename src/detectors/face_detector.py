import cv2
import mediapipe as mp
import numpy as np
from .base_detector import BaseDetector
from detector_types import FaceDetection
from typing import List, Optional, Dict
import time
from concurrent.futures import ThreadPoolExecutor

class FaceDetector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Optimize detection parameters for better accuracy
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Keep high-accuracy model
            min_detection_confidence=0.3  # Lowered from 0.5
        )
        
        # Optimize face mesh parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            min_detection_confidence=0.4,  # Lowered from 0.65
            min_tracking_confidence=0.4,   # Lowered from 0.65
            refine_landmarks=True  # Enable refined landmarks
        )
        
        # Rest of initialization...
        self._min_face_size = 20  # Lowered from 15 < 20
        self._max_face_size = 1000
        self._tracked_faces = []
        self._face_id_counter = 0
        self._last_detection_time = 0
        self._tracking_history = []
        self._max_history = 30
        self._max_tracking_age = 8  # More balanced tracking parameters
        self._tracking_threshold = 0.2  # Lowered from 0.3
        self._min_detection_interval = 1.0 / 60  # Increased detection frequency
        self._distance_weights = {
            'near': 1.0,
            'mid': 0.95,  # Increased from 0.9
            'far': 0.85   # Increased from 0.7
        }
        self.use_face_mesh = True
        
        # Add drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define landmark indices for key facial features
        self.FACE_FEATURES = {
            'nose': [
                # Bridge and tip of nose
                168, 6, 197, 195, 5, 4,
                # Nostrils
                242, 141, 94, 370, 462, 250
            ],
            'mouth': [
                # Outer lip
                61, 146, 91, 181, 84, 17,
                # Corner points
                314, 405, 321, 375,
                # Lower lip
                291, 409, 270, 269, 267, 0,
                # Upper lip
                37, 39, 40, 185, 61
            ],
            'face_oval': [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162
            ]
        }
        
        # Add drawing parameters for better visualization
        self.feature_colors = {
            'nose': (0, 255, 0),    # Green
            'mouth': (0, 0, 255),   # Red
            'face_oval': (255, 255, 255)  # White
        }
        
        self.line_thickness = {
            'nose': 1,
            'mouth': 1,
            'face_oval': 1
        }
        
        # Add pupil landmark indices
        self.FACE_FEATURES.update({
            'left_pupil': [468],   # Left eye pupil landmark
            'right_pupil': [473],  # Right eye pupil landmark
        })
        
        # Add iris landmarks for more precise tracking
        self.IRIS_LANDMARKS = {
            'left_iris': [468, 469, 470, 471, 472],
            'right_iris': [473, 474, 475, 476, 477]
        }
        
        # Update hair landmark indices to better match natural hair
        # self.HAIR_LANDMARKS = {
        #     'hairline': [
        #         # Forehead and temple points
        #         10, 108, 67, 109, 69, 151, 337, 299, 333, 297, 332,
        #         # Side points
        #         162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
        #         # Top of head points
        #         10, 338, 297
        #     ]
        # }
        
        # Reduce padding and adjust detection parameters
        self.hair_padding = 0.15  # Reduced from 0.4
        self.expansion_factor = 1.1  # Reduced from 1.2
        
        # Adjust color ranges for better detection of dark hair
        self.hair_color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),
            'dark_brown': ([0, 0, 30], [180, 255, 70]),
            'brown': ([0, 0, 50], [180, 255, 100]),
            'blonde': ([0, 0, 90], [180, 255, 200])
        }
        
        # Add detailed eye landmarks
        self.FACE_FEATURES.update({
            'left_eye': [
                # Upper eyelid
                33, 246, 161, 160, 159, 158, 157, 173,
                # Lower eyelid
                133, 155, 154, 153, 145, 144, 163, 7
            ],
            'right_eye': [
                # Upper eyelid
                362, 398, 384, 385, 386, 387, 388, 466,
                # Lower eyelid
                263, 249, 390, 373, 374, 380, 381, 382
            ],
            'left_eyebrow': [
                # Eyebrow points
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46
            ],
            'right_eyebrow': [
                # Eyebrow points
                300, 293, 334, 296, 336, 285, 295, 282, 283, 276
            ]
        })
        
        # Add eye-specific colors
        self.feature_colors.update({
            'left_eye': (255, 255, 0),     # Yellow
            'right_eye': (255, 255, 0),    # Yellow
            'left_eyebrow': (0, 255, 255), # Cyan
            'right_eyebrow': (0, 255, 255) # Cyan
        })
        
        # Add iris landmarks for more precise eye tracking
        self.IRIS_LANDMARKS = {
            'left_iris': [468, 469, 470, 471, 472],  # Left iris landmarks
            'right_iris': [473, 474, 475, 476, 477]  # Right iris landmarks
        }

    def process(self, frame: np.ndarray) -> List[FaceDetection]:
        """Process frame and detect multiple faces with hair detection."""
        try:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = processed_frame.shape[:2]
            
            # Process with face mesh
            mesh_results = self.face_mesh.process(processed_frame)
            faces = []
            
            if mesh_results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                    # Calculate bounding box from face landmarks
                    x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
                    y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
                    
                    x_min = int(min(x_coordinates) * frame_width)
                    y_min = int(min(y_coordinates) * frame_height)
                    x_max = int(max(x_coordinates) * frame_width)
                    y_max = int(max(y_coordinates) * frame_height)
                    
                    # Create bbox tuple (x, y, width, height)`
                    bbox = (
                        x_min,
                        y_min,
                        x_max - x_min,  # width
                        y_max - y_min   # height
                    )
                    
                    # Extract facial features with improved accuracy
                    facial_features = {}
                    for feature_name, indices in self.FACE_FEATURES.items():
                        points = []
                        for idx in indices:
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * frame_width)
                            y = int(landmark.y * frame_height)
                            points.append({
                                'x': x,
                                'y': y
                            })
                        facial_features[feature_name] = points
                    
                    # Calculate refined pupil positions using iris landmarks
                    pupils = self._calculate_pupil_positions(face_landmarks, frame_width, frame_height)
                    
                    # Add hair detection
                    hair_data = self._detect_hair(frame, face_landmarks, frame_width, frame_height)
                    
                    # Calculate eye features
                    eye_data = self._calculate_eye_features(face_landmarks, frame_width, frame_height)
                    
                    # Update facial features with eye data
                    facial_features.update(eye_data['eyes'])
                    
                    # Create face detection object
                    face = {
                        'id': len(faces),
                        'bbox': bbox,
                        'confidence': 1.0,
                        'facial_features': facial_features,
                        'iris': eye_data['iris'],
                        'pupils': eye_data['pupils'],
                        'hair': hair_data,
                        'has_mesh': True
                    }
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return []

    def _calculate_pupil_positions(self, face_landmarks, frame_width, frame_height):
        """Calculate precise pupil positions using iris landmarks."""
        pupils = {}
        
        # Calculate left pupil position
        left_iris_points = []
        for idx in self.IRIS_LANDMARKS['left_iris']:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            left_iris_points.append((x, y))
        
        # Calculate right pupil position
        right_iris_points = []
        for idx in self.IRIS_LANDMARKS['right_iris']:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            right_iris_points.append((x, y))
        
        # Calculate center points (pupils)
        if left_iris_points:
            left_x = sum(p[0] for p in left_iris_points) / len(left_iris_points)
            left_y = sum(p[1] for p in left_iris_points) / len(left_iris_points)
            pupils['left_pupil'] = {'x': int(left_x), 'y': int(left_y)}
        
        if right_iris_points:
            right_x = sum(p[0] for p in right_iris_points) / len(right_iris_points)
            right_y = sum(p[1] for p in right_iris_points) / len(right_iris_points)
            pupils['right_pupil'] = {'x': int(right_x), 'y': int(right_y)}
        
        return pupils

    def _update_face_tracking(self, detected_faces):
        """Enhanced face tracking with ID assignment and age tracking."""
        if not isinstance(self._tracked_faces, list):
            self._tracked_faces = []
        
        unmatched_detections = detected_faces.copy()
        new_tracks = []
        
        # Update existing tracks
        for track in self._tracked_faces:
            track['age'] = track.get('age', 0) + 1
            
            if track['age'] > self._max_tracking_age:
                continue
                
            best_match = None
            best_iou = self._tracking_threshold
            
            for face in unmatched_detections:
                iou = self._calculate_iou(track['bbox'], face['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = face
            
            if best_match:
                # Update track while preserving mesh data
                unmatched_detections.remove(best_match)
                track.update({
                    'bbox': best_match['bbox'],
                    'confidence': best_match['confidence'],
                    'raw_confidence': best_match.get('raw_confidence', best_match['confidence']),
                    'distance_category': best_match.get('distance_category', 'mid'),
                    'facial_features': best_match.get('facial_features', {}),  # Preserve facial features
                    'has_mesh': best_match.get('has_mesh', False),  # Preserve mesh flag
                    'age': 0
                })
                new_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for face in unmatched_detections:
            new_track = {
                'id': self._face_id_counter,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'raw_confidence': face.get('raw_confidence', face['confidence']),
                'distance_category': face.get('distance_category', 'mid'),
                'facial_features': face.get('facial_features', {}),  # Include facial features
                'has_mesh': face.get('has_mesh', False),  # Include mesh flag
                'age': 0
            }
            new_tracks.append(new_track)
            self._face_id_counter += 1
        
        self._tracked_faces = new_tracks
        return new_tracks

    def _predict_faces(self):
        """Predict face locations when skipping detection."""
        if not self._tracked_faces:
            return []
            
        predicted_faces = []
        current_time = time.time()
        
        for track in self._tracked_faces:
            if track['age'] <= self._max_tracking_age:
                predicted_face = track.copy()
                predicted_face['confidence'] *= max(0, 1 - (track['age'] / self._max_tracking_age))
                predicted_faces.append(predicted_face)
        
        return predicted_faces

    def _preprocess_frame(self, frame: np.ndarray, target_size: tuple, scale: float) -> np.ndarray:
        """Simplified preprocessing."""
        try:
            h, w = frame.shape[:2]
            target_h, target_w = target_size
            
            # Simple resize without additional processing
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _merge_detections(self, faces):
        """Improved merge logic with stricter criteria."""
        if not faces:
            return []
            
        merged = []
        used = set()
        
        # Sort faces by confidence and size
        faces = sorted(faces, 
                      key=lambda x: x['confidence'] * (x['bbox'][2] * x['bbox'][3]), 
                      reverse=True)
        
        for i, face1 in enumerate(faces):
            if i in used:
                continue
                
            if not self._validate_face_properties(face1):
                continue
                
            current_group = [face1]
            used.add(i)
            
            # Base IoU threshold
            base_iou = 0.6
            if face1['distance_category'] == 'far':
                base_iou = 0.5
            
            for j, face2 in enumerate(faces):
                if j in used:
                    continue
                    
                iou = self._calculate_iou(face1['bbox'], face2['bbox'])
                conf_diff = abs(face1['raw_confidence'] - face2['raw_confidence'])
                
                if iou > base_iou and conf_diff < 0.15:
                    current_group.append(face2)
                    used.add(j)
            
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                merged.append(self._weighted_merge_face_group(current_group))
        
        return merged

    def _weighted_merge_face_group(self, face_group):
        """Merge a group of overlapping face detections."""
        if not face_group:
            return None
            
        # Use highest confidence detection as base
        base_face = max(face_group, key=lambda x: x['confidence'])
        
        total_confidence = sum(face['confidence'] for face in face_group)
        weights = [face['confidence'] / total_confidence for face in face_group]
        
        # Weighted average of bounding boxes
        x = sum(face['bbox'][0] * w for face, w in zip(face_group, weights))
        y = sum(face['bbox'][1] * w for face, w in zip(face_group, weights))
        w = sum(face['bbox'][2] * w for face, w in zip(face_group, weights))
        h = sum(face['bbox'][3] * w for face, w in zip(face_group, weights))
        
        return {
            'bbox': (int(x), int(y), int(w), int(h)),
            'confidence': base_face['confidence'],
            'raw_confidence': base_face['raw_confidence'],
            'distance_category': base_face['distance_category']
        }

    def _validate_face_properties(self, face):
        """More lenient face validation."""
        bbox = face['bbox']
        aspect_ratio = bbox[2] / bbox[3]
        
        # More lenient aspect ratio
        if not 0.4 <= aspect_ratio <= 1.6:
            return False
            
        # More lenient confidence thresholds
        min_conf = {
            'near': 0.5,
            'mid': 0.55,
            'far': 0.6
        }
        
        if face['raw_confidence'] < min_conf[face['distance_category']]:
            return False
            
        # Size validation
        face_size = bbox[2] * bbox[3]
        if face_size < (self._min_face_size * self._min_face_size):
            return False
        if face_size > (self._max_face_size * self._max_face_size):
            return False
            
        return True

    def _estimate_distance(self, face_width_pixels):
        """Estimate distance to face using focal length formula."""
        return (self._focal_length * self._avg_face_width) / face_width_pixels

    def _calculate_quality_score(self, confidence, face_size, distance):
        """Enhanced quality scoring."""
        # Normalize size on a curve
        size_score = 1 - np.exp(-face_size / 20000)
        
        # Distance factor (inverse relationship)
        distance_factor = 1 / (1 + distance/20)
        
        # Weighted combination
        return (confidence * 0.4 + size_score * 0.3 + distance_factor * 0.3)

    def close(self):
        """Release resources."""
        self.detector.close()
        self.face_mesh.close()

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Optimized detection method with frame rate control."""
        if frame is None:
            return []
            
        current_time = time.time()
        if current_time - self._last_detection_time < self._min_detection_interval:
            return self._tracked_faces  # Return last tracked faces if skipping detection
            
        self._last_detection_time = current_time
        
        frame_h, frame_w = frame.shape[:2]
        all_faces = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for size, scale in zip(self._processing_sizes, self._scale_factors):
                futures.append(
                    executor.submit(
                        self._process_single_scale, 
                        frame, 
                        size, 
                        scale, 
                        frame_w, 
                        frame_h
                    )
                )
            
            for future in futures:
                faces = future.result()
                if faces:
                    all_faces.extend(faces)
        
        merged_faces = self._merge_detections(all_faces)
        return self._update_tracking(merged_faces)

    def _process_single_scale(self, frame, size, scale, frame_w, frame_h):
        """Helper method for parallel processing."""
        processed = self._preprocess_frame(frame, size, scale)
        results = self.detector.process(processed)
        
        if not results.detections:
            return []
            
        h, w = processed.shape[:2]
        scale_x = frame_w / w
        scale_y = frame_h / h
        
        return self._process_detections(
            results.detections, 
            frame_w, 
            frame_h, 
            scale_x, 
            scale_y, 
            self.detector.min_detection_confidence
        )

    def _process_detections(self, detections, frame_width, frame_height, scale_x, scale_y, min_conf):
        """Process detections with more lenient validation."""
        faces = []
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            confidence = detection.score[0]
            
            # More lenient confidence threshold
            if confidence < 0.3:  # Lowered from 0.5
                continue
                
            x = int(bbox.xmin * frame_width)
            y = int(bbox.ymin * frame_height)
            w = int(bbox.width * frame_width)
            h = int(bbox.height * frame_height)
            
            # More lenient size validation
            if w < self._min_face_size or h < self._min_face_size:
                continue
            if w > self._max_face_size or h > self._max_face_size:
                continue
                
            # More lenient aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < 0.4 or aspect_ratio > 1.6:  # Wider range
                continue
                
            face_size = w * h
            
            # More lenient distance categories
            if face_size > 20000:  # Lowered from 25000
                distance_category = 'near'
            elif face_size > 6000:  # Lowered from 8000
                distance_category = 'mid'
            else:
                distance_category = 'far'
                if confidence < 0.35:  # Lowered from 0.55
                    continue
            
            face = {
                'bbox': (x, y, w, h),
                'confidence': confidence * self._distance_weights[distance_category],
                'raw_confidence': confidence,
                'distance_category': distance_category
            }
            
            faces.append(face)
        
        return faces

    def _update_tracking(self, detected_faces):
        """Update face tracking information."""
        current_time = time.time()
        
        # Update tracking history
        if len(self._tracking_history) >= self._max_history:
            self._tracking_history.pop(0)
        self._tracking_history.append(self._tracked_faces.copy())
        
        # Update age of existing tracks
        for track in self._tracked_faces:
            track['age'] = track.get('age', 0) + 1
        
        # Remove old tracks
        self._tracked_faces = [
            track for track in self._tracked_faces 
            if track['age'] < self._max_tracking_age
        ]
        
        # Match new detections to existing tracks
        new_tracks = []
        unmatched_detections = detected_faces.copy()
        
        # Try to match with existing tracks first
        for track in self._tracked_faces:
            best_match = None
            best_iou = self._tracking_threshold
            
            for face in unmatched_detections:
                iou = self._calculate_iou(track['bbox'], face['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = face
            
            if best_match:
                # Update existing track
                unmatched_detections.remove(best_match)
                track.update({
                    'bbox': best_match['bbox'],
                    'confidence': best_match['confidence'],
                    'raw_confidence': best_match.get('raw_confidence', best_match['confidence']),
                    'distance_category': best_match.get('distance_category', 'mid'),
                    'age': 0,
                    'last_seen': current_time
                })
                new_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for face in unmatched_detections:
            new_tracks.append({
                'id': self._face_id_counter,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'raw_confidence': face.get('raw_confidence', face['confidence']),
                'distance_category': face.get('distance_category', 'mid'),
                'age': 0,
                'last_seen': current_time
            })
            self._face_id_counter += 1
        
        self._tracked_faces = new_tracks
        return new_tracks

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area

    def _draw_facial_features(self, frame, features):
        """Draw facial features on the frame."""
        colors = {
            'left_eye': (255, 0, 0),    # Blue
            'right_eye': (255, 0, 0),   # Blue
            'nose': (0, 255, 0),        # Green
            'mouth': (0, 0, 255),       # Red
            'left_ear': (255, 255, 0),  # Cyan
            'right_ear': (255, 255, 0), # Cyan
            'face_oval': (255, 255, 255)# White
        }
        
        for feature_name, points in features.items():
            color = colors.get(feature_name, (0, 255, 0))
            
            # Draw lines connecting the points
            for i in range(len(points)):
                pt1 = (points[i]['x'], points[i]['y'])
                pt2 = (points[(i + 1) % len(points)]['x'], points[(i + 1) % len(points)]['y'])
                cv2.line(frame, pt1, pt2, color, 1)
            
            # Draw points
            for point in points:
                cv2.circle(frame, (point['x'], point['y']), 1, color, -1)

    def toggle_face_mesh(self):
        """Toggle face mesh detection and visualization."""
        self.use_face_mesh = not self.use_face_mesh
        return self.use_face_mesh

    def process_frame(self, frame):
        try:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with face mesh
            face_mesh_results = self.face_mesh.process(rgb_frame)
            
            faces = []
            if face_mesh_results.multi_face_landmarks:
                height, width = frame.shape[:2]
                
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    # Get face bounding box
                    x_coordinates = [landmark.x for landmark in face_landmarks.landmark]
                    y_coordinates = [landmark.y for landmark in face_landmarks.landmark]
                    
                    x_min, x_max = min(x_coordinates), max(x_coordinates)
                    y_min, y_max = min(y_coordinates), max(y_coordinates)
                    
                    bbox = [
                        int(x_min * width),
                        int(y_min * height),
                        int((x_max - x_min) * width),
                        int((y_max - y_min) * height)
                    ]
                    
                    # Extract facial features
                    facial_features = {}
                    for feature_name, indices in self.FACE_FEATURES.items():
                        points = []
                        for idx in indices:
                            landmark = face_landmarks.landmark[idx]
                            points.append({
                                'x': int(landmark.x * width),
                                'y': int(landmark.y * height)
                            })
                        facial_features[feature_name] = points
                    
                    # Create face detection object
                    face = {
                        'id': len(faces),  # Simple ID assignment
                        'bbox': bbox,
                        'confidence': 1.0,  # Face mesh doesn't provide confidence
                        'facial_features': facial_features,
                        'has_mesh': True
                    }
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return []

    def _detect_hair(self, frame, face_landmarks, frame_width, frame_height):
        """Improved hair detection for natural hair."""
        try:
            # Get hairline points
            hairline_points = []
            for idx in self.HAIR_LANDMARKS['hairline']:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                hairline_points.append((x, y))

            # Calculate hair region with reduced padding
            x_coords = [p[0] for p in hairline_points]
            y_coords = [p[1] for p in hairline_points]
            
            min_x = max(0, min(x_coords) - int(frame_width * 0.05))  # Reduced side padding
            max_x = min(frame_width, max(x_coords) + int(frame_width * 0.05))
            min_y = max(0, min(y_coords))  # Start from actual hairline
            
            # Calculate more precise hair height
            actual_hair_height = int((max(y_coords) - min_y) * self.hair_padding)
            hair_top = max(0, min_y - actual_hair_height)
            
            # Create hair mask
            hair_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            
            # Use convex hull with reduced expansion
            hull_points = cv2.convexHull(np.array(hairline_points))
            center = np.mean(hull_points, axis=0)
            expanded_hull = []
            
            for point in hull_points:
                vector = point - center
                expanded_point = center + vector * self.expansion_factor
                expanded_hull.append(expanded_point)
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            
            # Draw the hair region
            cv2.fillPoly(hair_mask, [expanded_hull], 255)
            
            # Apply slight blur for smoother edges
            hair_mask = cv2.GaussianBlur(hair_mask, (3, 3), 0)
            
            return {
                'hair_region': {
                    'x': min_x,
                    'y': hair_top,
                    'width': max_x - min_x,
                    'height': max(y_coords) - hair_top + actual_hair_height
                },
                'hair_color': self._detect_hair_color(frame, hair_mask),
                'hairline_points': [{'x': x, 'y': y} for x, y in hairline_points]
            }
            
        except Exception as e:
            print(f"Error in hair detection: {str(e)}")
            return None

    def _detect_hair_color(self, hair_region, hair_mask):
        """Enhanced hair color detection with better handling of dark hair."""
        try:
            # Convert to HSV for better color detection
            hsv_region = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
            
            # Get pixels where mask is non-zero
            hair_pixels = hsv_region[hair_mask > 0]
            
            if len(hair_pixels) == 0:
                return 'unknown'
            
            # Calculate histogram of value channel
            v_channel = hair_pixels[:, 2]
            hist = np.histogram(v_channel, bins=50)[0]
            
            # Find dominant value range
            dominant_value = np.argmax(hist)
            
            # Simplified color classification based on value channel
            if dominant_value < 60:
                return 'black'
            elif dominant_value < 90:
                return 'dark_brown'
            elif dominant_value < 120:
                return 'brown'
            else:
                return 'blonde'
            
        except Exception as e:
            print(f"Error in hair color detection: {str(e)}")
            return 'unknown'

    def _calculate_eye_features(self, face_landmarks, frame_width, frame_height):
        """Calculate detailed eye features including iris and pupil positions."""
        eye_features = {}
        
        # Process each eye
        for eye_name in ['left_eye', 'right_eye']:
            points = []
            for idx in self.FACE_FEATURES[eye_name]:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                points.append({'x': x, 'y': y})
            eye_features[eye_name] = points
        
        # Process eyebrows
        for brow_name in ['left_eyebrow', 'right_eyebrow']:
            points = []
            for idx in self.FACE_FEATURES[brow_name]:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                points.append({'x': x, 'y': y})
            eye_features[brow_name] = points
        
        # Calculate iris positions
        iris_features = {}
        for iris_name, indices in self.IRIS_LANDMARKS.items():
            points = []
            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                points.append({'x': x, 'y': y})
            iris_features[iris_name] = points
        
        # Calculate pupil positions (center of iris)
        pupils = {}
        for side in ['left', 'right']:
            iris_points = iris_features[f'{side}_iris']
            if iris_points:
                x_avg = sum(p['x'] for p in iris_points) / len(iris_points)
                y_avg = sum(p['y'] for p in iris_points) / len(iris_points)
                pupils[f'{side}_pupil'] = {'x': int(x_avg), 'y': int(y_avg)}
        
        return {
            'eyes': eye_features,
            'iris': iris_features,
            'pupils': pupils
        }

    # Define facial landmark indices
    FACIAL_LANDMARKS = {
        'left_eye': [33, 133, 160, 159, 158, 144, 145, 153],
        'right_eye': [362, 263, 387, 386, 385, 373, 374, 380],
        'nose': [1, 2, 98, 327],
        'mouth': [61, 291, 0, 17, 57, 287],
        'left_ear': [127, 234, 93, 132],
        'right_ear': [356, 454, 323, 361],
        'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162]
    }