import os
import cv2
import pickle
import sys
import pprint
from pathlib import Path
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from handLM_metric import calculate_hand_accuracy, calculate_accuracy_percentage
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Paths
NEAR_INFRARED_DIR = Path("/home/user/Desktop/RP/datasets/multimodalkaggle/near-infrared")
HL_RESULTS_DIR = Path("/home/user/Desktop/RP/datasets/multimodalkaggle/HLResults_multimodal")
VISUALIZATIONS_DIR = Path("/home/user/Desktop/RP/datasets/multimodalkaggle/visualizations")
GROUND_TRUTH_DIR = HL_RESULTS_DIR  # Assuming ground truth is stored here

# Ensure output directories exist
HL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

def visualize_landmarks(image, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS):
    """Draws hand landmarks and connections on the image using MediaPipe's drawing utilities."""
    annotated_image = image.copy()
    
    # Convert list of landmarks to NormalizedLandmarkList
    if isinstance(hand_landmarks[0], list):  # If we have a list of hand landmarks
        for landmarks in hand_landmarks:
            landmark_proto = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in landmarks
                ]
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                landmark_proto,
                connections,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
    else:  # If we have a single hand's landmarks
        landmark_proto = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in hand_landmarks
            ]
        )
        mp_drawing.draw_landmarks(
            annotated_image,
            landmark_proto,
            connections,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
    
    return annotated_image

def build_ground_truth_map(ground_truth_dir):
    """
    Pre-builds a mapping from unique_id to ground truth file.

    Args:
        ground_truth_dir (Path): Directory containing ground truth files.

    Returns:
        dict: A dictionary mapping unique_id (str) to Path of the ground truth file.
    """
    ground_truth_map = {}
    for gt_file in ground_truth_dir.iterdir():
        if gt_file.is_file():
            unique_id = gt_file.stem  # If there's no extension, stem is the filename
            ground_truth_map[unique_id] = gt_file
            print(f"Mapping unique_id '{unique_id}' to '{gt_file.name}'")
    print(f"Total ground truth entries mapped: {len(ground_truth_map)}")
    return ground_truth_map

def extract_unique_id(image_filepath, near_infrared_dir):
    """
    Generates a unique_id for the image based on its relative path, excluding the handedness suffix.

    Args:
        image_filepath (Path): Path object of the image file.
        near_infrared_dir (Path): Base directory of near-infrared images.

    Returns:
        str: Generated unique_id matching the ground truth file naming convention.
    """
    # Generate relative path from near_infrared_dir
    relative_path = image_filepath.relative_to(near_infrared_dir)
    
    # Extract directory parts
    directory_parts = list(relative_path.parent.parts)
    
    # Extract the frame number without the handedness suffix
    # Assuming the image filename is in the format 'frame_<number>_<handedness>.png'
    frame_stem = image_filepath.stem  # e.g., 'frame_4312_l'
    frame_parts = frame_stem.rsplit('_', 1)  # Split into ['frame_4312', 'l']
    
    if len(frame_parts) == 2:
        frame_number = frame_parts[0]  # 'frame_4312'
    else:
        # If the filename does not follow the expected format, handle accordingly
        print(f"Unexpected image filename format: {image_filepath.name}")
        frame_number = frame_stem  # Fallback to the entire stem
    
    # Construct the unique_id by combining directory parts and frame number
    unique_id_parts = directory_parts + [frame_number]
    unique_id = "_".join(unique_id_parts)
    
    return unique_id

def main():
    # Build ground truth mapping
    ground_truth_map = build_ground_truth_map(GROUND_TRUTH_DIR)
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,         # Set to True for image mode
        max_num_hands=2,                # Maximum number of hands to detect
        min_detection_confidence=0.5    # Minimum confidence for detection
    ) as hands:

        # Iterate through the near-infrared dataset
        for subject_dir in NEAR_INFRARED_DIR.iterdir():
            if subject_dir.is_dir():
                for gesture_dir in subject_dir.iterdir():
                    if gesture_dir.is_dir():
                        for sample_dir in gesture_dir.iterdir():
                            if sample_dir.is_dir():
                                for image_file in sample_dir.glob("*.png"):
                                    # Process each image
                                    print(f"Processing image: {image_file}")

                                    # Generate unique_id
                                    unique_id = extract_unique_id(image_file, NEAR_INFRARED_DIR)
                                    print(f"Generated unique_id: {unique_id}")

                                    # Find corresponding ground truth file
                                    ground_truth_file = ground_truth_map.get(unique_id)
                                    if not ground_truth_file:
                                        print(f"Ground truth not found for image: {image_file}")
                                        continue

                                    # Read the image
                                    image = cv2.imread(str(image_file))
                                    if image is None:
                                        print(f"Failed to read image: {image_file}")
                                        continue

                                    # Convert the BGR image to RGB for MediaPipe
                                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                                    # Run MediaPipe Hands
                                    results = hands.process(image_rgb)

                                    # Extract detected hand landmarks
                                    hand_landmarks = results.multi_hand_landmarks
                                    handedness = results.multi_handedness

                                    if hand_landmarks and handedness:
                                        # Convert detected landmarks to NormalizedLandmarkList
                                        predicted_landmarks = []
                                        for hand in hand_landmarks:
                                            landmark_list = []
                                            for lm in hand.landmark:
                                                landmark = landmark_pb2.NormalizedLandmark(
                                                    x=lm.x,
                                                    y=lm.y,
                                                    z=lm.z
                                                )
                                                landmark_list.append(landmark)
                                            landmark_proto = landmark_pb2.NormalizedLandmarkList(
                                                landmark=landmark_list
                                            )
                                            predicted_landmarks.append(landmark_proto)

                                        # Load corresponding ground truth
                                        with open(ground_truth_file, 'rb') as f:
                                            ground_truth = pickle.load(f)

                                        # Ensure ground truth is in the expected format
                                        ground_truth_landmarks = []
                                        for hand in ground_truth.hand_landmarks:
                                            landmark_list = []
                                            for lm in hand:
                                                landmark = landmark_pb2.NormalizedLandmark(
                                                    x=lm.x,
                                                    y=lm.y,
                                                    z=lm.z
                                                )
                                                landmark_list.append(landmark)
                                            landmark_proto = landmark_pb2.NormalizedLandmarkList(
                                                landmark=landmark_list
                                            )
                                            ground_truth_landmarks.append(landmark_proto)

                                        # Calculate accuracy
                                        try:
                                            mean_distance = calculate_hand_accuracy(predicted_landmarks, ground_truth_landmarks)
                                            accuracy_pct = calculate_accuracy_percentage(mean_distance)
                                            print(f"Mean Euclidean Distance: {mean_distance:.4f}")
                                            print(f"Accuracy: {accuracy_pct:.2f}%")
                                        except ValueError as e:
                                            print(f"Error calculating accuracy: {e}")
                                            
                                            # Detailed Information about Ground Truth
                                            print(f"\n--- Ground Truth Landmarks {len(ground_truth_landmarks)}---")
                                            pprint.pprint(ground_truth_landmarks)
                                            
                                            # Detailed Information about Predicted Results
                                            print(f"\n--- Predicted Landmarks {len(predicted_landmarks)}---")
                                            pprint.pprint(predicted_landmarks)
                                            
                                            # Terminate the program
                                            sys.exit(1)

                                        # Visualize landmarks
                                        visualized_image = visualize_landmarks(image, hand_landmarks)

                                        # Save visualization
                                        visualization_path = VISUALIZATIONS_DIR / f"{image_file.stem}_vis.png"
                                        cv2.imwrite(str(visualization_path), visualized_image)
                                        print(f"Saved visualization to: {visualization_path}")
                                    else:
                                        print(f"No hand landmarks detected in image: {image_file}")

                                    # Optionally, remove or adjust the following line to process all images
                                    # return

if __name__ == "__main__":
    main()