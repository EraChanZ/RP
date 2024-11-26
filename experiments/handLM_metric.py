import math
from typing import List
import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from hand_landmarker_result import HandLandmarkerResultCustom
from mediapipe.framework.formats import landmark_pb2

def calculate_euclidean_distance(point1, point2) -> float:
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

def calculate_hand_accuracy(
    predicted_landmarks: List[List[NormalizedLandmark]],
    ground_truth_landmarks: List[List[NormalizedLandmark]]
) -> float:
    if not all(isinstance(hl, landmark_pb2.NormalizedLandmarkList) for hl in predicted_landmarks):
        raise TypeError("predicted_landmarks must be a list of NormalizedLandmarkList objects")

    if not all(isinstance(hl, landmark_pb2.NormalizedLandmarkList) for hl in ground_truth_landmarks):
        raise TypeError("ground_truth_landmarks must be a list of NormalizedLandmarkList objects")

    if len(predicted_landmarks) != len(ground_truth_landmarks):
        raise ValueError("Number of hands in prediction and ground truth do not match.")

    total_distance = 0.0
    total_landmarks = 0

    for pred_hand, gt_hand in zip(predicted_landmarks, ground_truth_landmarks):
        print(f"Pred Hand: {len(pred_hand.landmark)}")
        print(f"GT Hand: {len(gt_hand.landmark)}")
        if len(pred_hand.landmark) != len(gt_hand.landmark):
            raise ValueError("Number of landmarks in prediction and ground truth hands do not match.")
        
        for pred_landmark, gt_landmark in zip(pred_hand.landmark, gt_hand.landmark):
            distance = calculate_euclidean_distance(pred_landmark, gt_landmark)
            total_distance += distance
            total_landmarks += 1

    if total_landmarks == 0:
        raise ValueError("No landmarks to compare.")

    mean_distance = total_distance / total_landmarks
    return mean_distance

def calculate_accuracy_percentage(mean_distance: float, threshold: float = 0.05) -> float:
    accuracy = max(0.0, (threshold - mean_distance) / threshold) * 100
    return accuracy

