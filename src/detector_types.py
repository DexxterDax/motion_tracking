from typing import TypedDict, List, Tuple, Dict, Optional
import numpy as np

class BoundingBox(TypedDict):
    x: int
    y: int
    w: int
    h: int

class FaceDetection(TypedDict):
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: List[Tuple[int, int]]

class HandDetection(TypedDict):
    landmarks: List[Tuple[int, int]]
    handedness: str
    confidence: float
    connections: List[Tuple[int, int]]
    id: int
    is_pointing: bool
    is_peace: bool
    is_thumbs_up: bool
    is_fist: bool
    is_open_palm: bool
    is_pinch: bool
    index_tip: Tuple[int, int]
    bbox: Tuple[int, int, int, int]

class PoseDetection(TypedDict):
    landmarks: List[Tuple[int, int]]
    connections: List[Tuple[int, int]]
    visibility: List[float]

class ProcessingStats(TypedDict):
    processed_frames: int
    dropped_frames: int
    avg_latency: float