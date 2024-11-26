from dataclasses import dataclass
from typing import List
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark
from mediapipe.tasks.python.components.containers.category import Category

@dataclass
class HandLandmarkerResultCustom:
    """Custom HandLandmarkerResult to handle missing data gracefully."""
    handedness: List[List[Category]]
    hand_landmarks: List[List[NormalizedLandmark]]
    hand_world_landmarks: List[List[Landmark]] 