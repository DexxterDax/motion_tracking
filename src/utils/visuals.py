import cv2
import numpy as np
from typing import List, Dict, Optional, Any
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

class Visualizer:
    def __init__(self):
        self.colors = {
            'face': (0, 255, 0),    # Green
            'hand': (255, 0, 0),    # Blue
            'pose': (0, 0, 255),    # Red
            'left_eye': (255, 255, 0),    # Yellow
            'right_eye': (255, 255, 0),   # Yellow
            'nose': (0, 255, 0),        # Green
            'mouth': (0, 0, 255),       # Red
            'left_ear': (255, 255, 0),  # Cyan
            'right_ear': (255, 255, 0), # Cyan
            'face_oval': (255, 255, 255), # White
            'left_pupil': (0, 255, 255),  # Yellow
            'right_pupil': (0, 255, 255),  # Yellow
            'hair': (165, 42, 42),  # Brown
            'hairline': (255, 255, 255),  # White
            'left_eyebrow': (0, 255, 255), # Cyan
            'right_eyebrow': (0, 255, 255),# Cyan
            'iris': (0, 255, 255),         # Cyan
            'pupil': (0, 255, 0)           # Green
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_thickness = 2
        self.font_scale = 0.5
        
        # Add smoothing parameters
        self.smooth_factor = 0.5  # Higher = more smoothing, but more lag (0.5-0.8 recommended)
        self._prev_bbox_by_id = {}  # Store previous bbox per face
        self._smoothed_features_by_id = {}  # Store smoothed features per face

    def _smooth_points_for_face(self, current_points, face_id, feature_name):
        """Apply smoothing to points for a specific face."""
        if not current_points:
            return current_points
            
        if face_id not in self._smoothed_features_by_id:
            self._smoothed_features_by_id[face_id] = {}
            
        if feature_name not in self._smoothed_features_by_id[face_id]:
            self._smoothed_features_by_id[face_id][feature_name] = current_points
            return current_points
            
        smoothed = []
        prev_points = self._smoothed_features_by_id[face_id][feature_name]
        
        if len(prev_points) != len(current_points):
            return current_points
            
        for i, current in enumerate(current_points):
            prev = prev_points[i]
            smoothed.append({
                'x': int(prev['x'] * self.smooth_factor + current['x'] * (1 - self.smooth_factor)),
                'y': int(prev['y'] * self.smooth_factor + current['y'] * (1 - self.smooth_factor))
            })
            
        self._smoothed_features_by_id[face_id][feature_name] = smoothed
        return smoothed

    def draw_results(self, frame, faces=None, poses=None, stats=None):
        """Draw all detections on the frame."""
        if frame is None:
            return None
            
        output_frame = frame.copy()
        
        # Clean up old face data
        if faces:
            current_face_ids = {face['id'] for face in faces if 'id' in face}
            # Remove smoothing data for faces that are no longer visible
            self._smoothed_features_by_id = {
                k: v for k, v in self._smoothed_features_by_id.items() 
                if k in current_face_ids
            }
            self._prev_bbox_by_id = {
                k: v for k, v in self._prev_bbox_by_id.items() 
                if k in current_face_ids
            }
        else:
            self._smoothed_features_by_id.clear()
            self._prev_bbox_by_id.clear()
        
        # Draw all faces
        if faces:
            for face in faces:
                self._draw_face(output_frame, face)
        
        return output_frame

    def _draw_performance_stats(self, frame, stats):
        """Draw performance statistics on frame."""
        try:
            height = frame.shape[0]
            
            # Convert stats to float if they're strings
            fps = float(stats.get('fps', 0))
            latency = float(stats.get('latency', 0))
            dropped = int(stats.get('dropped', 0))
            
            stats_lines = [
                f"FPS: {fps:.1f}",
                f"Latency: {latency:.1f}ms",
                f"Dropped: {dropped}"
            ]
            
            y_offset = height - (len(stats_lines) * 25)
            for i, line in enumerate(stats_lines):
                y = y_offset + (i * 20)
                cv2.putText(
                    frame,
                    line,
                    (10, y),
                    self.font,
                    self.font_scale,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                
        except Exception as e:
            print(f"Error drawing stats: {str(e)}")

    def _draw_face(self, frame, face):
        """Draw face detection results with improved eye visualization."""
        try:
            if not isinstance(face, dict) or 'id' not in face:
                return
                
            face_id = face['id']
            
            # Draw bounding box with per-face smoothing
            if 'bbox' in face:
                x, y, w, h = face['bbox']
                
                # Smooth bbox coordinates for this face
                if face_id in self._prev_bbox_by_id:
                    prev_bbox = self._prev_bbox_by_id[face_id]
                    x = int(prev_bbox[0] * self.smooth_factor + x * (1 - self.smooth_factor))
                    y = int(prev_bbox[1] * self.smooth_factor + y * (1 - self.smooth_factor))
                    w = int(prev_bbox[2] * self.smooth_factor + w * (1 - self.smooth_factor))
                    h = int(prev_bbox[3] * self.smooth_factor + h * (1 - self.smooth_factor))
                self._prev_bbox_by_id[face_id] = [x, y, w, h]
                
                # Draw rectangle and ID
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face'], 1)
                cv2.putText(frame, f"Face {face_id}", (x, y - 10), self.font, 0.5, self.colors['face'], 1, cv2.LINE_AA)
            
            # Draw facial features
            if 'facial_features' in face:
                features = face['facial_features']
                
                # Draw each feature
                for feature_name, points in features.items():
                    if len(points) < 2:
                        continue
                        
                    color = self.colors.get(feature_name, (0, 255, 0))
                    
                    # Draw connected lines for continuous features
                    if feature_name in ['face_oval', 'mouth']:
                        # Create closed loop
                        pts = np.array([[p['x'], p['y']] for p in points], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
                    else:
                        # Draw connected segments
                        for i in range(len(points) - 1):
                            pt1 = (points[i]['x'], points[i]['y'])
                            pt2 = (points[i + 1]['x'], points[i + 1]['y'])
                            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

            # Draw eyes
            for feature_name in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow']:
                if feature_name in face['facial_features']:
                    points = face['facial_features'][feature_name]
                    color = self.colors[feature_name]
                    
                    # Draw eye contour
                    pts = np.array([[p['x'], p['y']] for p in points], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

            # Draw iris
            if 'iris' in face:
                for iris_name, points in face['iris'].items():
                    pts = np.array([[p['x'], p['y']] for p in points], np.int32)
                    cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, 
                                self.colors['iris'], 1, cv2.LINE_AA)

            # Draw pupils
            if 'pupils' in face:
                for pupil_name, point in face['pupils'].items():
                    cv2.circle(frame, (point['x'], point['y']), 
                             2, self.colors['pupil'], -1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error drawing face: {str(e)}")

    def _draw_pose(self, frame, pose):
        """Draw pose detection results."""
        try:
            if not isinstance(pose, dict):
                return
                
            if 'landmarks' in pose and pose['landmarks']:
                # Draw pose landmarks
                for landmark in pose['landmarks']:
                    if len(landmark) == 2:  # Ensure landmark has x,y coordinates
                        cv2.circle(
                            frame,
                            (int(landmark[0]), int(landmark[1])),
                            3,
                            self.colors['pose'],
                            -1,
                            cv2.LINE_AA
                        )
                
                # Draw connections if available
                if 'connections' in pose and pose['connections']:
                    landmarks = pose['landmarks']
                    for connection in pose['connections']:
                        if len(connection) == 2:
                            start_idx, end_idx = connection
                            if (start_idx < len(landmarks) and 
                                end_idx < len(landmarks)):
                                start_point = landmarks[start_idx]
                                end_point = landmarks[end_idx]
                                if len(start_point) == 2 and len(end_point) == 2:
                                    cv2.line(
                                        frame,
                                        (int(start_point[0]), int(start_point[1])),
                                        (int(end_point[0]), int(end_point[1])),
                                        self.colors['pose'],
                                        self.line_thickness,
                                        cv2.LINE_AA
                                    )
                                    
        except Exception as e:
            print(f"Error drawing pose: {str(e)}")