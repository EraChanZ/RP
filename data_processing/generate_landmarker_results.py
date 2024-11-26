import os
import xml.etree.ElementTree as ET
from typing import List, Dict
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm  # For progress bar
import pickle  # For serialization
from concurrent.futures import ThreadPoolExecutor, as_completed

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark
from mediapipe.tasks.python.components.containers.category import Category

def setup_logging(debug_log_path: str = 'debug.log', max_bytes: int = 10**7, backup_count: int = 5):
    """
    Configures logging with separate handlers for console and debug file.

    Args:
        debug_log_path (str): Path to the debug log file.
        max_bytes (int): Maximum size of the debug log file before rotation.
        backup_count (int): Number of backup files to keep.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels; individual handlers manage output

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler for INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler for DEBUG level with rotation
    file_handler = RotatingFileHandler(debug_log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

class SkeletalToLandmarkerConverter:
    """
    A converter class to transform skeletal XML data into HandLandmarkerResult objects.
    """

    def __init__(self, skeletal_dir: str, image_width: int, image_height: int):
        """
        Initializes the converter with the skeletal data directory and image dimensions.

        Args:
            skeletal_dir (str): Root directory of skeletal XML files.
            image_width (int): Width of the associated images.
            image_height (int): Height of the associated images.
        """
        self.skeletal_dir = skeletal_dir
        self.image_width = image_width
        self.image_height = image_height

    def generate_results(self) -> Dict[str, HandLandmarkerResult]:
        """
        Parses skeletal XML files and converts them into HandLandmarkerResult objects.

        Returns:
            Dict[str, HandLandmarkerResult]: Mapping from unique ID to result objects.
        """
        results = {}
        xml_files = []

        # Traverse all subdirectories to find XML files
        for root_dir, dirs, files in os.walk(self.skeletal_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))

        logging.info(f"Found {len(xml_files)} XML files in {self.skeletal_dir} and its subdirectories.")

        if not xml_files:
            logging.warning("No XML files found. Please check the directory path and file extensions.")

        processed_count = 0
        N = 1000  # Log every Nth iteration

        for file_path in tqdm(xml_files, desc="Processing skeletal XML files"):
            # Generate unique_id based on relative path
            relative_path = os.path.relpath(file_path, self.skeletal_dir)
            unique_id = relative_path.replace('.xml', '').replace(os.sep, '_')

            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Initialize lists for HandLandmarkerResult
                handedness = []
                hand_landmarks = []
                hand_world_landmarks = []

                # Navigate to the Right hand data
                right_image = root.find('.//RigthImage')  # Note: 'RigthImage' as per the XML example
                if right_image is None:
                    logging.warning(f"No RigthImage found in {file_path}. Skipping.")
                    continue

                hands = right_image.find('Hands')
                if hands is None:
                    logging.warning(f"No Hands found in {file_path}. Skipping.")
                    continue

                right_hand = hands.find('Right')
                if right_hand is None:
                    logging.warning(f"No Right hand data found in {file_path}. Skipping.")
                    continue

                # Populate handedness (assuming only right hand)
                handedness.append(Category(
                    index=0,
                    score=1.0,
                    category_name='Right',
                    display_name='Right'
                ))

                # Initialize lists for landmarks
                finger_landmarks = []
                finger_world_landmarks = []

                # Add WRIST landmark (landmark 0) using hand center
                center = right_hand.find('Center')
                if center is None:
                    logging.warning(f"No Center position found in {file_path}. Skipping.")
                    continue

                center_coords = self._parse_opencv_matrix(center)
                wrist_landmark = NormalizedLandmark(
                    x=center_coords[0] / self.image_width,
                    y=center_coords[1] / self.image_height,
                    z=center_coords[2]
                )
                finger_landmarks.append(wrist_landmark)
                finger_world_landmarks.append(Landmark(
                    x=center_coords[0],
                    y=center_coords[1],
                    z=center_coords[2]
                ))

                # Process fingers in MediaPipe order
                finger_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                # Note: MediaPipe uses this order for each finger: MCP, PIP, DIP, TIP
                joint_order = ['mcpPosition', 'pipPosition', 'dipPosition', 'TipPosition']

                fingers = right_hand.find('Fingers')
                if fingers is None:
                    logging.warning(f"No Fingers element found in {file_path}. Skipping.")
                    continue

                for finger_name in finger_order:
                    finger = fingers.find(finger_name)
                    if finger is None:
                        logging.warning(f"{finger_name} data missing in {file_path}.")
                        # Add placeholder landmarks to maintain correct ordering
                        for _ in range(4):
                            finger_landmarks.append(NormalizedLandmark(x=0, y=0, z=0))
                            finger_world_landmarks.append(Landmark(x=0, y=0, z=0))
                        continue

                    for joint in joint_order:
                        joint_elem = finger.find(joint)
                        if joint_elem is None:
                            logging.warning(f"{joint} missing for {finger_name} in {file_path}")
                            # Add placeholder landmark
                            finger_landmarks.append(NormalizedLandmark(x=0, y=0, z=0))
                            finger_world_landmarks.append(Landmark(x=0, y=0, z=0))
                            continue

                        coords = self._parse_opencv_matrix(joint_elem)
                        finger_landmarks.append(NormalizedLandmark(
                            x=coords[0] / self.image_width,
                            y=coords[1] / self.image_height,
                            z=coords[2]
                        ))
                        finger_world_landmarks.append(Landmark(
                            x=coords[0],
                            y=coords[1],
                            z=coords[2]
                        ))

                # Verify we have exactly 21 landmarks (1 wrist + 4 joints × 5 fingers)
                if len(finger_landmarks) == 21:
                    hand_landmarks.append(finger_landmarks)
                    hand_world_landmarks.append(finger_world_landmarks)
                    result = HandLandmarkerResult(
                        handedness=handedness,
                        hand_landmarks=hand_landmarks,
                        hand_world_landmarks=hand_world_landmarks
                    )
                    results[unique_id] = result
                else:
                    logging.warning(f"Incorrect number of landmarks ({len(finger_landmarks)}) in {file_path}")

                processed_count += 1

                # Log every Nth iteration
                if processed_count % N == 0:
                    logging.info(f"Processed {processed_count}/{len(xml_files)} files: {unique_id}")

            except KeyError as e:
                logging.error(f"Failed to process {file_path}: Missing key {e}")
            except ET.ParseError as e:
                logging.error(f"Failed to process {file_path}: XML parsing error {e}")
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")

        logging.info(f"Completed processing {processed_count} files.")
        return results

    def _parse_opencv_matrix(self, matrix_element) -> List[float]:
        """
        Parses an OpenCV matrix from the XML and returns the data as a list of floats.

        Args:
            matrix_element: XML element containing the matrix data.

        Returns:
            List[float]: Parsed matrix data.
        """
        data_text = matrix_element.find('data').text.strip()
        data = [float(num) for num in data_text.split()]
        return data

def main():
    setup_logging(debug_log_path='debug.log', max_bytes=10**7, backup_count=5)
    skeletal_dir = '/home/user/Desktop/RP/datasets/multimodalkaggle/skeletal'
    image_width = 640  # Example width, adjust as needed
    image_height = 480  # Example height, adjust as needed

    converter = SkeletalToLandmarkerConverter(skeletal_dir, image_width, image_height)
    landmarker_results = converter.generate_results()

    # Define the output directory
    output_dir = '/home/user/Desktop/RP/datasets/multimodalkaggle/HLResults_multimodal'
    os.makedirs(output_dir, exist_ok=True)

    def save_result(unique_id, result):
        """
        Serializes and saves a HandLandmarkerResult instance.
        """
        filename = unique_id.replace('/', '_').replace('\\', '_').replace('.xml', '.pkl')
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(result, f)
            logging.debug(f"Saved HandLandmarkerResult to {file_path}")
            return True, unique_id
        except Exception as e:
            logging.error(f"Failed to save HandLandmarkerResult for {unique_id}: {e}")
            return False, unique_id

    # Define the number of worker threads
    MAX_WORKERS = 8  # Adjust based on your system's capabilities

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all save tasks
        future_to_id = {executor.submit(save_result, uid, res): uid for uid, res in landmarker_results.items()}

        for future in as_completed(future_to_id):
            success, uid = future.result()
            if success:
                pass  # Optionally, track successes
            else:
                pass  # Optionally, handle failures

    logging.info(f"All HandLandmarkerResult instances have been saved to {output_dir}.")

if __name__ == "__main__":
    main()