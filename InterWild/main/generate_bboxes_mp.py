import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = r"C:\Users\vladi\RP\InterWild\main\hand_landmarker.task"

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_image = mp.Image.create_from_file(r'C:\Users\vladi\RP\Research\IR_videos\every60thframe\012332998_frame_0.jpg')
# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
with HandLandmarker.create_from_options(options) as landmarker:
    hand_landmarker_result = landmarker.detect(mp_image)
    print(hand_landmarker_result)

    