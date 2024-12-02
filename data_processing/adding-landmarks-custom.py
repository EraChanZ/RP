import cv2
import json
import os
import numpy as np

# List to store the points for all images
all_landmarks = []
left_hand_landmarks = []  # Temporary list to store left hand landmarks
right_hand_landmarks = []  # Temporary list to store right hand landmarks
extra_point_left = None
extra_point_right = None

# Resize factor
resize_factor = 0.5  # Change this to adjust the size of the displayed image

# Number of landmarks for one hand
NUM_LANDMARKS = 21

# Function to capture mouse click events
def click_event(event, x, y, flags, param):
    global left_hand_landmarks, right_hand_landmarks, image_width, image_height, extra_point_right, extra_point_left
    if event == cv2.EVENT_LBUTTONDOWN:
        # Normalize the (x, y) coordinates
        print(f"Clicked: ({x}, {y})")

        normalized_x = x / (image_width * resize_factor)
        normalized_y = y / (image_height * resize_factor)

        print(f"Normalized: ({normalized_x}, {normalized_y})")

        # Add the normalized point to the current hand landmarks
        if len(left_hand_landmarks) < NUM_LANDMARKS:
            left_hand_landmarks.append({"x": normalized_x, "y": normalized_y})
            print(f"Left hand: {len(left_hand_landmarks)} landmarks captured.")
        elif len(right_hand_landmarks) < NUM_LANDMARKS:
            right_hand_landmarks.append({"x": normalized_x, "y": normalized_y})
            print(f"Right hand: {len(right_hand_landmarks)} landmarks captured.")
        elif extra_point_left is None:
            extra_point_left = {"x": normalized_x, "y": normalized_y}
            print("Extra point for left hand captured.")
        elif extra_point_right is None:
            extra_point_right = {"x": normalized_x, "y": normalized_y}
            print("Extra point for right hand captured.")

        # Display the point on the image
        cv2.circle(image_resized, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Image', image_resized)

        # Once we have 21 points for each hand, signal completion
        if len(left_hand_landmarks) == NUM_LANDMARKS and len(right_hand_landmarks) == NUM_LANDMARKS:
            print("Landmarks for both hands captured.")

# List of specific images to annotate
image_files = [
    "IMG20241127105250.jpg",
    "IMG20241127105256.jpg",
    "IMG20241127105346.jpg",
    "IMG20241127102901.jpg",
    "IMG20241127102854.jpg",
    "IMG20241127102811.jpg",
    "IMG20241127102805.jpg",
    "IMG20241127102649.jpg",
    "IMG20241127102643.jpg",
    "IMG20241127102538.jpg",
    "IMG20241127102532.jpg",
    "IMG20241127102430.jpg",
    "IMG20241127102423.jpg",
    "IMG20241127102317.jpg",
    "IMG20241127102216.jpg",
    "IMG20241127102205.jpg"
]

# Directory paths
input_directory = "/home/user/Desktop/RP/datasets/ourData/Data/Dataset/IR images"
output_file = "/home/user/Desktop/RP/datasets/ourData/Data/Dataset/Annotations/landmarks_normalized.json"

# Add these functions after the imports
def enhance_image(image):
    # Increase brightness
    brightened = cv2.convertScaleAbs(image, alpha=2.0, beta=0)  # alpha=2.0 doubles brightness
    
    # Apply CLAHE
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

# Process each image
for image_file in image_files:
    current_image_name = image_file
    image_path = os.path.join(input_directory, image_file)

    while True:
        # Reset landmarks for the current image
        left_hand_landmarks = []
        right_hand_landmarks = []
        extra_point_left = None
        extra_point_right = None

        # Load and enhance the image
        image = cv2.imread(image_path)
        image = enhance_image(image)
        image_height, image_width, _ = image.shape

        # Resize the image to fit the window (if necessary)
        image_resized = cv2.resize(image, (int(image_width * resize_factor), int(image_height * resize_factor)))

        # Set up the window and mouse callback
        cv2.imshow("Image", image_resized)
        cv2.setMouseCallback("Image", click_event)

        # Wait for the user to click on all the landmarks
        print(f"Annotating {image_file}. Click {NUM_LANDMARKS} points for each hand.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Confirm whether to proceed or retry
        decision = input(f"Finished {image_file}. Type 'n' to proceed to the next image, or 'r' to retry: ").strip().lower()

        if decision == 'n':
            # Remove any previously saved landmarks for this image
            all_landmarks = [item for item in all_landmarks if item["image"] != current_image_name]

            # Save the current landmarks as a list of lists for left and right hands
            all_landmarks.append({
                "image": current_image_name,
                "landmarks": [left_hand_landmarks, right_hand_landmarks],
                "extra_points": {
                    "left_hand": extra_point_left,
                    "right_hand": extra_point_right,
                }
            })
            print(f"Landmarks for {image_file} saved.")
            break
        elif decision == 'r':
            print(f"Retrying {image_file}. Previous landmarks will be overridden.")

# Save all landmarks to a JSON file with a more readable structure
with open(output_file, 'w') as f:
    json.dump(all_landmarks, f, indent=4)

print(f"Annotations saved to '{output_file}'")