#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:27:08 2024

@author: Irene
"""
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import shutil    
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' # Fill in your own path to tesseract.exe, differs per computer 
import math
import matplotlib.gridspec as gridspec
import time
import pandas as pd

#%% To start 

start_time = time.time()

# Ask user to select a video file
file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("mp4 files", "*.mp4"), ("All files", "*.mov")])

# Extract the video name and extension
video_name, video_extension = os.path.splitext(os.path.basename(file_path))

# Determine the parent directory of the 'Videos' folder
parent_directory = os.path.dirname(os.path.dirname(file_path))

# Specify the directory path for the 'Output' folder
output_path = os.path.join(parent_directory, '3. Output', video_name)

# Check if the output folder already exists
if os.path.exists(output_path):
    # Find a unique name for the output folder
    suffix = 1
    while True:
        unique_output_path = f"{output_path} ({suffix})"
        if not os.path.exists(unique_output_path):
            output_path = unique_output_path
            break
        suffix += 1

# Create the output folder
os.makedirs(output_path, exist_ok=True)

#%% Trim video and save separate 'before cooling' and 'after cooling' videos

# Create a tkinter window (it won't be shown)
root = tk.Tk()
root.withdraw()
cap = cv2.VideoCapture(file_path)
fps = cap.get(cv2.CAP_PROP_FPS)
length2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
window_name = 'Adjust the trackbars to set the correct starting and end points, then press a key to close the window'
non_empty_frames = []

def find_last_non_empty_frame():
    last_non_empty_frame = None
    for i in range(length2 - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        err, frame = cap.read()
        if frame is not None:
            last_non_empty_frame = i
            break
    return last_non_empty_frame

length = find_last_non_empty_frame() # Find the position of the last non-empty frame. Sometimes the last frames in the video are empty. 

def onchange(trackbarvalue):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarvalue)
    err, frame = cap.read()    
    # Rotate the frame 90 degrees to the right
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)    
    cv2.imshow(window_name, frame)

def save_rotated_frame(image, video_output_folder, filename):
    rotated_frame = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    save_path = os.path.join(output_path, filename)
    cv2.imwrite(save_path, rotated_frame)
    
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.createTrackbar('Start - before cooling', window_name, 0, length, onchange)
cv2.createTrackbar('End - before cooling', window_name, int(length/16), length, onchange)
cv2.createTrackbar('Start - after cooling', window_name, int(length/4), length, onchange)
cv2.createTrackbar('End - after cooling', window_name, length, length, onchange)

# Set the window size (otherwise it will result in an error)
window_width = 800
window_height = 600
cv2.resizeWindow(window_name, window_width, window_height)

onchange(0)
cv2.waitKey(0)

start_before_cooling = cv2.getTrackbarPos('Start - before cooling', window_name)
end_before_cooling = cv2.getTrackbarPos('End - before cooling', window_name)
start_after_cooling = cv2.getTrackbarPos('Start - after cooling', window_name) 
end_after_cooling = cv2.getTrackbarPos('End - after cooling', window_name) 
        
# Save images with frame numbers
cap.set(cv2.CAP_PROP_POS_FRAMES, start_before_cooling)
err, frame = cap.read()
save_rotated_frame(frame, output_path, f'A1. Before cooling - start - frame {start_before_cooling}.jpg')
cap.set(cv2.CAP_PROP_POS_FRAMES, end_before_cooling)
err, frame = cap.read()
save_rotated_frame(frame, output_path, f'A2. Before cooling - end - frame {end_before_cooling}.jpg')
cap.set(cv2.CAP_PROP_POS_FRAMES, start_after_cooling)
err, frame = cap.read()
save_rotated_frame(frame, output_path, f'A3. After cooling - start - frame {start_after_cooling}.jpg')
cap.set(cv2.CAP_PROP_POS_FRAMES, end_after_cooling)
err, frame = cap.read()
save_rotated_frame(frame, output_path, f'A4. After cooling - end - frame {end_after_cooling}.jpg')

cv2.destroyAllWindows()
cv2.waitKey(1)

# Get frame rate of the original video
original_frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Set the video capture object to the selected start point for the first trimmed video (before cooling)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_before_cooling)

# Read the frames and write them to a new video file for the first trimmed video (before cooling)
before_cooling_video_path = os.path.join(output_path, f"{video_name}_before_cooling{video_extension}")

# Check if the trimmed video already exists, and delete it if it does
if os.path.exists(before_cooling_video_path):
    os.remove(before_cooling_video_path)
    print("Existing before cooling video deleted.")

# Create a new trimmed video for the first trimmed video (before cooling)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_before_cooling = cv2.VideoWriter(before_cooling_video_path, fourcc, original_frame_rate, (int(cap.get(3)), int(cap.get(4)))) # frame rate 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        out_before_cooling.write(frame)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_before_cooling:
            break
    else:
        break

# Release the video writer object for the first trimmed video (before cooling)
out_before_cooling.release()

# Set the video capture object to the selected start point for the second trimmed video (after cooling)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_after_cooling)

# Read the frames and write them to a new video file for the second trimmed video (after cooling)
after_cooling_video_path = os.path.join(output_path, f"{video_name}_after_cooling{video_extension}")

# Check if the trimmed video already exists, and delete it if it does
if os.path.exists(after_cooling_video_path):
    os.remove(after_cooling_video_path)
    print("Existing after cooling video deleted.")

# Create a new trimmed video for the second trimmed video (after cooling)
out_after_cooling = cv2.VideoWriter(after_cooling_video_path, fourcc, original_frame_rate, (int(cap.get(3)), int(cap.get(4)))) # frame rate 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        out_after_cooling.write(frame)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_after_cooling:
            break
    else:
        break

# Release the video writer object for the second trimmed video (after cooling)
out_after_cooling.release()

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

#%% MediaPipe landmark tracking for the two videos

# Initialize MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

# Function to find the last non-empty frame
def find_last_non_empty_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_non_empty_frame = None
    for i in range(length - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        err, frame = cap.read()
        if frame is not None:
            last_non_empty_frame = i
            break
    cap.release()
    return last_non_empty_frame

# Function to perform landmark tracking and store coordinates
def perform_landmark_tracking(video_path, before=True, total_frames=None):
    # Initialize MediaPipe Hands module
    hands = mphands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          model_complexity=1,
                          min_detection_confidence=0,
                          min_tracking_confidence=0.5)

    # Open the video file for capturing frames
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = find_last_non_empty_frame(video_path)

    # Create an empty list to store landmark coordinates
    num_landmarks = 21  # Since there are 21 landmarks per hand
    
    left_x_coords = np.full((total_frames, num_landmarks), np.nan)
    left_y_coords = np.full((total_frames, num_landmarks), np.nan)
    left_z_coords = np.full((total_frames, num_landmarks), np.nan)

    right_x_coords = np.full((total_frames, num_landmarks), np.nan)
    right_y_coords = np.full((total_frames, num_landmarks), np.nan)
    right_z_coords = np.full((total_frames, num_landmarks), np.nan)
    
    frame_index = 0

    while cap.isOpened():
        # Read a frame from the video
        success, image = cap.read()

        # Rotate the frame 90 degrees clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Break the loop if the video is finished
        if not success:
            break

        # Process the frame using the MediaPipe Hands module
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #left_detected = False
        #right_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Identify left or right hand by the handedness
                if handedness.classification[0].label == 'Right':
                    x_coords = left_x_coords
                    y_coords = left_y_coords
                    z_coords = left_z_coords
                    #left_detected = True
                elif handedness.classification[0].label == 'Left':
                    x_coords = right_x_coords
                    y_coords = right_y_coords
                    z_coords = right_z_coords
                    #right_detected = True
                for i, landmark in enumerate(hand_landmarks.landmark):
                    # Accessing landmark coordinates
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    # Update landmark coordinates in the arrays
                    x_coords[frame_index, i] = x
                    y_coords[frame_index, i] = y
                    z_coords[frame_index, i] = z
                    
        frame_index += 1
        
        # Exit the loop if the last frame is reached
        if total_frames is not None and frame_index >= total_frames:
            break

     # Transpose the landmark coordinates arrays
    left_x_coords = left_x_coords.T
    left_y_coords = left_y_coords.T
    left_z_coords = left_z_coords.T

    right_x_coords = right_x_coords.T
    right_y_coords = right_y_coords.T
    right_z_coords = right_z_coords.T

    # Release the video capture object
    cap.release()

    if before:
        return left_x_coords, left_y_coords, left_z_coords, right_x_coords, right_y_coords, right_z_coords
    else:
        return left_x_coords, left_y_coords, left_z_coords, right_x_coords, right_y_coords, right_z_coords

# Get the total number of frames for the before cooling video
cap_before = cv2.VideoCapture(before_cooling_video_path)
before_length2 = int(cap_before.get(cv2.CAP_PROP_FRAME_COUNT))
before_length = find_last_non_empty_frame(before_cooling_video_path)
cap_before.release()

# Get the total number of frames for the after cooling video
cap_after = cv2.VideoCapture(after_cooling_video_path)
after_length2 = int(cap_after.get(cv2.CAP_PROP_FRAME_COUNT))
after_length = find_last_non_empty_frame(after_cooling_video_path)
cap_after.release()

# Perform landmark tracking for the before cooling video
before_left_x, before_left_y, before_left_z, before_right_x, before_right_y, before_right_z = perform_landmark_tracking(before_cooling_video_path, before=True)

# Perform landmark tracking for the after cooling video
after_left_x, after_left_y, after_left_z, after_right_x, after_right_y, after_right_z = perform_landmark_tracking(after_cooling_video_path, before=False)

# Count the number of NaN rows in landmark tracking
num_nan_before_left = sum(np.isnan(before_left_x).all(axis=0))
num_nan_before_right = sum(np.isnan(before_right_x).all(axis=0))
num_nan_after_left = sum(np.isnan(after_left_x).all(axis=0))
num_nan_after_right = sum(np.isnan(after_right_x).all(axis=0))

#%% Save frames with Mediapipe landmarks

# Create the "Test frames landmarks" folder within the output_path
test_frames_landmarks_folder = os.path.join(output_path, "B. Frames with MediaPipe landmarks")
os.makedirs(test_frames_landmarks_folder, exist_ok=True)

# Determine frame numbers for the first frame, last frame, and in-between frames for after cooling video
first_frame_number_after = 0
last_frame_number_after = after_length - 1  # 'after_length' is the total number of frames in the video
in_between_frame_numbers_after = np.linspace(first_frame_number_after, last_frame_number_after, num=45, dtype=int)

# Open the after cooling video file
cap_after = cv2.VideoCapture(after_cooling_video_path)

# Check if the after cooling video file opened successfully
if not cap_after.isOpened():
    print("Error: Unable to open after cooling video file")
    exit()

# Loop over the frame numbers and draw landmarks on the frames for after cooling video
for frame_number in [first_frame_number_after] + list(in_between_frame_numbers_after) + [last_frame_number_after]:
    # Set the frame position to the desired frame
    cap_after.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap_after.read()
    if not success:
        print(f"Error: Unable to read frame {frame_number} from after cooling video")
        continue

    # Rotate the frame 90 degrees clockwise
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert the BGR image to RGB for plotting with matplotlib
    frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)

    # Define the landmark coordinates for the frame for the left hand
    left_landmark_x_norm = after_left_x[:, frame_number]
    left_landmark_y_norm = after_left_y[:, frame_number]

    # Define the landmark coordinates for the frame for the right hand
    right_landmark_x_norm = after_right_x[:, frame_number]
    right_landmark_y_norm = after_right_y[:, frame_number]

    # Get the height and width of the frame
    height, width, _ = frame_rgb.shape

    # Convert normalized coordinates to absolute pixel coordinates for the left hand
    left_landmark_x = left_landmark_x_norm * width
    left_landmark_y = left_landmark_y_norm * height

    # Convert normalized coordinates to absolute pixel coordinates for the right hand
    right_landmark_x = right_landmark_x_norm * width
    right_landmark_y = right_landmark_y_norm * height

    # Calculate time in seconds
    time_sec = frame_number / fps

    # Plot the frame
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title(f"After cooling frame {frame_number} (time: {time_sec:.2f} s)")

    # Plot left hand landmarks as red dots
    for x, y in zip(left_landmark_x, left_landmark_y):
        plt.plot(x, y, 'ro', markersize=3)

    # Plot right hand landmarks as blue dots
    for x, y in zip(right_landmark_x, right_landmark_y):
        plt.plot(x, y, 'bo', markersize=3)

    plt.axis("off")

    # Save the frame as an image in the "Test frames landmarks" folder
    frame_output_path = os.path.join(test_frames_landmarks_folder, f"2. After cooling_frame {frame_number}.jpg")
    plt.savefig(frame_output_path)

    # Close the plot to prevent memory leaks
    plt.close()

# Release the after cooling video capture object
cap_after.release()

# Determine frame numbers for the first frame, last frame, and in-between frames for the before cooling video
first_frame_number_before = 0
last_frame_number_before = before_length - 1  # 'before_length' is the total number of frames in the before cooling video
in_between_frame_numbers_before = np.linspace(first_frame_number_before, last_frame_number_before, num=5, dtype=int)

# Open the before cooling video file
cap_before = cv2.VideoCapture(before_cooling_video_path)

# Check if the before cooling video file opened successfully
if not cap_before.isOpened():
    print("Error: Unable to open before cooling video file")
    exit()

# Loop over the first and last frame numbers for the before cooling video
for frame_number in [first_frame_number_before] + list(in_between_frame_numbers_before) + [last_frame_number_before]:
    # Set the frame position to the desired frame for the before cooling video
    cap_before.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap_before.read()
    if not success:
        print(f"Error: Unable to read frame {frame_number}")
        continue

    # Rotate the frame 90 degrees clockwise
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert the BGR image to RGB for plotting with matplotlib
    frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)

    # Define the landmark coordinates for the frame for the left hand
    left_landmark_x_norm = before_left_x[:, frame_number]
    left_landmark_y_norm = before_left_y[:, frame_number]

    # Define the landmark coordinates for the frame for the right hand
    right_landmark_x_norm = before_right_x[:, frame_number]
    right_landmark_y_norm = before_right_y[:, frame_number]

    # Get the height and width of the frame
    height, width, _ = frame_rgb.shape

    # Convert normalized coordinates to absolute pixel coordinates for the left hand
    left_landmark_x = left_landmark_x_norm * width
    left_landmark_y = left_landmark_y_norm * height

    # Convert normalized coordinates to absolute pixel coordinates for the right hand
    right_landmark_x = right_landmark_x_norm * width
    right_landmark_y = right_landmark_y_norm * height

    # Calculate time in seconds
    time_sec = frame_number / fps

    # Plot the frame
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Before cooling frame {frame_number} (time: {time_sec:.2f} s)")

    # Plot left hand landmarks as red dots
    for x, y in zip(left_landmark_x, left_landmark_y):
        plt.plot(x, y, 'ro', markersize=3)

    # Plot right hand landmarks as blue dots
    for x, y in zip(right_landmark_x, right_landmark_y):
        plt.plot(x, y, 'bo', markersize=3)

    plt.axis("off")

    # Save the frame as an image in the "Test frames landmarks" folder
    frame_output_path = os.path.join(test_frames_landmarks_folder, f"1. Before cooling_frame {frame_number}.jpg")
    plt.savefig(frame_output_path)

    # Close the plot to prevent memory leaks
    plt.close()

# Release the before cooling video capture object
cap_before.release()

#%% Create ROI matrices

# Define the row indices
row_indices = [4, 8, 6, 20, 18]  # Select regions of interest

# Define a dictionary to map the landmark indices to their names
hand_landmarks = {
    4: "thumb tip",
    6: "index finger pip",
    8: "index finger tip",
    18: "pinky pip",
    20: "pinky tip"
}

# Initialize an empty list to store the rows
combined_after_ROIs_x = []
combined_after_ROIs_y = []
combined_before_ROIs_x = []
combined_before_ROIs_y = []
labels = []

# Concatenate the specified rows from left and right hand data and create labels
for row_index in row_indices:
    combined_after_ROIs_x.extend([after_left_x[row_index, :], after_right_x[row_index, :]])
    combined_after_ROIs_y.extend([after_left_y[row_index, :], after_right_y[row_index, :]])
    combined_before_ROIs_x.extend([before_left_x[row_index, :], before_right_x[row_index, :]])
    combined_before_ROIs_y.extend([before_left_y[row_index, :], before_right_y[row_index, :]])

# Compute the palm points for all frames
def compute_palm_point_left(wrist_x, wrist_y, point17_x, point17_y):
    palm_x = wrist_x + abs(point17_x - wrist_x) / 1.5
    palm_y = wrist_y - abs(point17_y - wrist_y) / 3
    return palm_x, palm_y

def compute_palm_point_right(wrist_x, wrist_y, point17_x, point17_y):
    palm_x = wrist_x - abs(point17_x - wrist_x) / 1.5
    palm_y = wrist_y - abs(point17_y - wrist_y) / 3
    return palm_x, palm_y

def compute_palm_point_left_upsidedown(wrist_x, wrist_y, point17_x, point17_y):
    palm_x = wrist_x - abs(point17_x - wrist_x) / 3.0
    palm_y = wrist_y + abs(point17_y - wrist_y) / 3.0
    return palm_x, palm_y

def compute_palm_point_right_upsidedown(wrist_x, wrist_y, point17_x, point17_y):
    palm_x = wrist_x + abs(point17_x - wrist_x) / 3.0
    palm_y = wrist_y + abs(point17_y - wrist_y) / 3.0
    return palm_x, palm_y

# For after data
left_wrist_x, left_wrist_y = after_left_x[0, :], after_left_y[0, :]
right_wrist_x, right_wrist_y = after_right_x[0, :], after_right_y[0, :]
left_point17_x, left_point17_y = after_left_x[17, :], after_left_y[17, :]
right_point17_x, right_point17_y = after_right_x[17, :], after_right_y[17, :]

# Check if the hands are upside down for both hands
if left_wrist_y[0] < left_point17_y[0] and right_wrist_y[0] < right_point17_y[0]:
    # If upside down for both hands, use alternative calculation for both hands
    compute_palm_left = compute_palm_point_left_upsidedown
    compute_palm_right = compute_palm_point_right_upsidedown
else:
    # If not upside down for both hands, use normal calculation for both hands
    compute_palm_left = compute_palm_point_left
    compute_palm_right = compute_palm_point_right

# Compute palm points for both hands (only once per video)
left_palm_x, left_palm_y = compute_palm_left(left_wrist_x, left_wrist_y, left_point17_x, left_point17_y)
right_palm_x, right_palm_y = compute_palm_right(right_wrist_x, right_wrist_y, right_point17_x, right_point17_y)

# Extend the palm rows for after data
combined_after_ROIs_x.extend([left_palm_x, right_palm_x])
combined_after_ROIs_y.extend([left_palm_y, right_palm_y])

# For before data
left_wrist_x, left_wrist_y = before_left_x[0, :], before_left_y[0, :]
right_wrist_x, right_wrist_y = before_right_x[0, :], before_right_y[0, :]
left_point17_x, left_point17_y = before_left_x[17, :], before_left_y[17, :]
right_point17_x, right_point17_y = before_right_x[17, :], before_right_y[17, :]

# Check if the hands are upside down for both hands
if left_wrist_y[0] < left_point17_y[0] and right_wrist_y[0] < right_point17_y[0]:
    # If upside down for both hands, use alternative calculation for both hands
    compute_palm_left = compute_palm_point_left_upsidedown
    compute_palm_right = compute_palm_point_right_upsidedown
else:
    # If not upside down for both hands, use normal calculation for both hands
    compute_palm_left = compute_palm_point_left
    compute_palm_right = compute_palm_point_right

# Compute palm points for both hands (only once per video)
left_palm_x, left_palm_y = compute_palm_left(left_wrist_x, left_wrist_y, left_point17_x, left_point17_y)
right_palm_x, right_palm_y = compute_palm_right(right_wrist_x, right_wrist_y, right_point17_x, right_point17_y)

# Extend the palm rows for before data
combined_before_ROIs_x.extend([left_palm_x, right_palm_x])
combined_before_ROIs_y.extend([left_palm_y, right_palm_y])

# Convert the lists of rows to NumPy arrays
after_ROIs_x = np.vstack(combined_after_ROIs_x)
after_ROIs_y = np.vstack(combined_after_ROIs_y)
before_ROIs_x = np.vstack(combined_before_ROIs_x)
before_ROIs_y = np.vstack(combined_before_ROIs_y)

# Extract the labels
labels = []
for row_index in row_indices:
    labels.append(f"Left {hand_landmarks[row_index]}")
    labels.append(f"Right {hand_landmarks[row_index]}")

# Add the labels for the palm points
labels.append("Left palm")
labels.append("Right palm")

#%% Coordinate filtering and plotting of x and y coordinates

# Define custom colors and labels
#custom_colors = ['#afc7e8', '#335d7e',  # Pair 1 (Blue)
                 #'#edb9c9', '#93485f',  # Pair 2 (Pink)
                 #'#b3dab0', '#488f4f',  # Pair 3 (Green)
                 #'#ea9393', '#9d2f2f',  # Pair 4 (Red)
                 #'#94c7de', '#447b94',  # Pair 5 (Turquoise)
                 #'#eda86c', '#b87023']  # Pair 6 (Yellow)

custom_colors = ['#ff9999', '#cc0000',  # Pair 1 (Light Red, Dark Red)
                 '#99ccff', '#003366',  # Pair 2 (Light Blue, Dark Blue)
                 '#99ff99', '#009933',  # Pair 3 (Light Green, Dark Green)
                 '#ffcc99', '#cc6600',  # Pair 4 (Light Orange, Dark Orange)
                 '#cc99ff', '#660099',  # Pair 5 (Light Purple, Dark Purple)
                 '#ffff99', '#999900']  # Pair 6 (Light Yellow, Dark Yellow)

labels = ['Left thumb tip', 'Right thumb tip', 'Left index finger tip', 'Right index finger tip', 
          'Left index finger pip', 'Right index finger pip', 'Left pinky tip', 'Right pinky tip', 
          'Left pinky pip', 'Right pinky pip', 'Left palm', 'Right palm']


# Create the "Landmark coordinate plots" folder within the output_path
landmarks_coordinate_plots_folder = os.path.join(output_path, "C. ROI coordinate plots")
os.makedirs(landmarks_coordinate_plots_folder, exist_ok=True)

# Number of frames
num_frames_after = after_ROIs_x.shape[1]
timesteps_after = np.arange(num_frames_after) / 30  # Assuming 30 FPS

# Number of frames for before_ROIs matrices
num_frames_before = before_ROIs_x.shape[1]
timesteps_before = np.arange(num_frames_before) / 30  # Assuming 30 FPS

# Function to remove outliers using the IQR method
def remove_outliers_iqr(column, factor):
    Q1 = np.nanpercentile(column, 25)
    Q3 = np.nanpercentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    filtered_column = column.copy()
    outliers = (column < lower_bound) | (column > upper_bound)
    filtered_column[outliers] = np.nan
    return filtered_column

# Function to interpolate NaN values
def interpolate_nan_values(column):
    column_series = pd.Series(column)
    interpolated_column = column_series.interpolate(method='linear', limit_direction='both').to_numpy()
    return interpolated_column

# Function for moving average
def moving_average(data, window_size):
    smoothed_data = np.copy(data)  # Make a copy of the data
    half_window = window_size // 2

    for i in range(len(data)):
        # Define the window range
        window_start = max(0, i - half_window)
        window_end = min(len(data), i + half_window + 1)

        # Calculate the mean of non-NaN values within the window
        window_values = data[window_start:window_end]
        non_nan_values = window_values[~np.isnan(window_values)]

        if len(non_nan_values) > 0:
            smoothed_data[i] = np.mean(non_nan_values)
        else:
            smoothed_data[i] = np.nan

    return smoothed_data

# Define window sizes for moving average
window_size_before = 10  # Window size for before_ROIs data
window_size_after = 1000  # Window size for after_ROIs data

# Lists to store preprocessed and smoothed X and Y coordinates
before_ROIs_x_filtered = []
before_ROIs_y_filtered = []
after_ROIs_x_filtered = []
after_ROIs_y_filtered = []

# Define factors for IQR filtering
factor_first = 0
factor_second = 1.5

# Iterate through each landmark (row) for before_ROIs matrices
for i in range(before_ROIs_x.shape[0]):
    # Preprocess X coordinates with first IQR filtering
    preprocessed_x = remove_outliers_iqr(before_ROIs_x[i], factor_first)
    interpolated_x = interpolate_nan_values(preprocessed_x)
    # Apply second IQR filtering
    preprocessed_x = remove_outliers_iqr(interpolated_x, factor_second)
    interpolated_x = interpolate_nan_values(preprocessed_x)
    # Apply moving average to X coordinates with window_size_before
    smoothed_x = moving_average(interpolated_x, window_size_before)
    before_ROIs_x_filtered.append(smoothed_x)
    
    # Preprocess Y coordinates with first IQR filtering
    preprocessed_y = remove_outliers_iqr(before_ROIs_y[i], factor_first)
    interpolated_y = interpolate_nan_values(preprocessed_y)
    # Apply second IQR filtering
    preprocessed_y = remove_outliers_iqr(interpolated_y, factor_second)
    interpolated_y = interpolate_nan_values(preprocessed_y)
    # Apply moving average to Y coordinates with window_size_before
    smoothed_y = moving_average(interpolated_y, window_size_before)
    before_ROIs_y_filtered.append(smoothed_y)

# Convert smoothed X and Y coordinates of before_ROIs to NumPy arrays
before_ROIs_x_filtered = np.array(before_ROIs_x_filtered)
before_ROIs_y_filtered = np.array(before_ROIs_y_filtered)

# Iterate through each landmark (row) for after_ROIs matrices
for i in range(after_ROIs_x.shape[0]):
    # Preprocess X coordinates with first IQR filtering
    preprocessed_x = remove_outliers_iqr(after_ROIs_x[i], factor_first)
    interpolated_x = interpolate_nan_values(preprocessed_x)
    # Apply second IQR filtering
    preprocessed_x = remove_outliers_iqr(interpolated_x, factor_second)
    interpolated_x = interpolate_nan_values(preprocessed_x)
    # Apply moving average to X coordinates with window_size_after
    smoothed_x = moving_average(interpolated_x, window_size_after)
    after_ROIs_x_filtered.append(smoothed_x)
    
    # Preprocess Y coordinates with first IQR filtering
    preprocessed_y = remove_outliers_iqr(after_ROIs_y[i], factor_first)
    interpolated_y = interpolate_nan_values(preprocessed_y)
    # Apply second IQR filtering
    preprocessed_y = remove_outliers_iqr(interpolated_y, factor_second)
    interpolated_y = interpolate_nan_values(preprocessed_y)
    # Apply moving average to Y coordinates with window_size_after
    smoothed_y = moving_average(interpolated_y, window_size_after)
    after_ROIs_y_filtered.append(smoothed_y)
    
# Convert smoothed X and Y coordinates of after_ROIs to NumPy arrays
after_ROIs_x_filtered = np.array(after_ROIs_x_filtered)
after_ROIs_y_filtered = np.array(after_ROIs_y_filtered)

# Get y-axis limits for after cooling x and x filtered
fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_after, after_ROIs_x[i], color=custom_colors[i])
y_limits_x_after = ax.get_ylim()
plt.close()

fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_after, after_ROIs_x_filtered[i], color=custom_colors[i])
y_limits_x_filtered_after = ax.get_ylim()
plt.close()

# Set consistent y-axis limits for after cooling x and x filtered
y_limits_x_after_combined = (min(y_limits_x_after[0], y_limits_x_filtered_after[0]), 
                             max(y_limits_x_after[1], y_limits_x_filtered_after[1]))

# Get y-axis limits for after cooling y and y filtered
fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_after, after_ROIs_y[i], color=custom_colors[i])
y_limits_y_after = ax.get_ylim()
plt.close()

fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_after, after_ROIs_y_filtered[i], color=custom_colors[i])
y_limits_y_filtered_after = ax.get_ylim()
plt.close()

# Set consistent y-axis limits for after cooling y and y filtered
y_limits_y_after_combined = (min(y_limits_y_after[0], y_limits_y_filtered_after[0]), 
                             max(y_limits_y_after[1], y_limits_y_filtered_after[1]))

# Get y-axis limits for before cooling x and x filtered
fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_before, before_ROIs_x[i], color=custom_colors[i])
y_limits_x_before = ax.get_ylim()
plt.close()

fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_before, before_ROIs_x_filtered[i], color=custom_colors[i])
y_limits_x_filtered_before = ax.get_ylim()
plt.close()

# Set consistent y-axis limits for before cooling x and x filtered
y_limits_x_before_combined = (min(y_limits_x_before[0], y_limits_x_filtered_before[0]), 
                              max(y_limits_x_before[1], y_limits_x_filtered_before[1]))

# Get y-axis limits for before cooling y and y filtered
fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_before, before_ROIs_y[i], color=custom_colors[i])
y_limits_y_before = ax.get_ylim()
plt.close()

fig, ax = plt.subplots()
for i in range(12):
    ax.plot(timesteps_before, before_ROIs_y_filtered[i], color=custom_colors[i])
y_limits_y_filtered_before = ax.get_ylim()
plt.close()

# Set consistent y-axis limits for before cooling y and y filtered
y_limits_y_before_combined = (min(y_limits_y_before[0], y_limits_y_filtered_before[0]), 
                              max(y_limits_y_before[1], y_limits_y_filtered_before[1]))

# Plot X coordinates of each landmark over time for after cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_after, after_ROIs_x[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('After cooling x coordinates of 12 ROIs', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Xx coordinate [0-1]')
ax.set_ylim(y_limits_x_after_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '2a. After cooling x coordinates.png'), bbox_inches='tight')
plt.close()

# Plot X filtered coordinates of each landmark over time for after cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_after, after_ROIs_x_filtered[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('After cooling x coordinates of 12 ROIs - filtered', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('x coordinate [0-1]')
ax.set_ylim(y_limits_x_after_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '2b. After cooling x coordinates - filtered.png'), bbox_inches='tight')
plt.close()

# Plot Y coordinates of each landmark over time for after cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_after, after_ROIs_y[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('After cooling y coordinates of 12 ROIs', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('y coordinate [0-1]')
ax.set_ylim(y_limits_y_after_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '2c. After cooling y coordinates.png'), bbox_inches='tight')
plt.close()

# Plot Y filtered coordinates of each landmark over time for after cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_after, after_ROIs_y_filtered[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('After cooling y coordinates of 12 ROIs - filtered', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('y coordinate [0-1]')
ax.set_ylim(y_limits_y_after_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '2d. After cooling y coordinates - filtered.png'), bbox_inches='tight')
plt.close()

# Plot X coordinates of each landmark over time for before cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_before, before_ROIs_x[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('Before cooling x coordinates of 12 ROIs', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('x coordinate [0-1]')
ax.set_ylim(y_limits_x_before_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '1a. Before cooling x coordinates.png'), bbox_inches='tight')
plt.close()

# Plot X filtered coordinates of each landmark over time for before cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_before, before_ROIs_x_filtered[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('Before cooling x coordinates of 12 ROIs - filtered', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('x coordinate [0-1]')
ax.set_ylim(y_limits_x_before_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '1b. Before cooling x coordinates - filtered.png'), bbox_inches='tight')
plt.close()

# Plot Y coordinates of each landmark over time for before cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_before, before_ROIs_y[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('Before cooling y coordinates of 12 ROIs', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('y coordinate [0-1]')
ax.set_ylim(y_limits_y_before_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '1c. Before cooling y coordinates.png'), bbox_inches='tight')
plt.close()

# Plot Y filtered coordinates of each landmark over time for before cooling
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(12):
    line_style = '-' if i % 2 == 0 else '--'
    ax.plot(timesteps_before, before_ROIs_y_filtered[i], label=labels[i], linestyle=line_style, color=custom_colors[i])

ax.set_title('Before cooling y coordinates of 12 ROIs - filtered', fontweight='bold')
ax.set_xlabel('Time [sec]')
ax.set_ylabel('y coordinate [0-1]')
ax.set_ylim(y_limits_y_before_combined)
ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
ax.grid(True)
plt.savefig(os.path.join(landmarks_coordinate_plots_folder, '1d. Before cooling y coordinates - filtered.png'), bbox_inches='tight')
plt.close()


#%% Save frames with ROIs

# Create function to draw circles around landmarks with specified color
def draw_circles(image, landmark_x, landmark_y, radius, color):
    for x, y in zip(landmark_x, landmark_y):
        if np.isnan(x) or np.isnan(y):
            continue  # Skip drawing if x or y is NaN
        cv2.circle(image, (int(x), int(y)), radius, color, 2)

radius = 10

# Create output directory if it does not exist
output_folder = os.path.join(output_path, "D. Frames with annotated ROIs")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(after_cooling_video_path)

# Determine frame numbers for the first frame, last frame, and in-between frames for the after cooling video
first_frame_number_after = 0
last_frame_number_after = after_length - 1  # 'after_length' is the total number of frames in the after cooling video
in_between_frame_numbers_after = np.linspace(first_frame_number_after, last_frame_number_after, num=45, dtype=int)

# Loop over the frame numbers and draw landmarks on the frames for the after cooling video
for frame_number in [first_frame_number_after] + list(in_between_frame_numbers_after) + [last_frame_number_after]:
    # Set the frame position to the desired frame for the after cooling video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()
    if not success:
        print(f"Error: Unable to read frame {frame_number}")
        continue

    # Rotate the frame 90 degrees clockwise
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert the BGR image to RGB for plotting with matplotlib
    frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)

    # Define the landmark coordinates for the frame for the left hand
    landmark_x_norm = after_ROIs_x_filtered[:, frame_number]
    landmark_y_norm = after_ROIs_y_filtered[:, frame_number]

    # Get the height and width of the frame
    height, width, _ = frame_rgb.shape

    # Convert normalized coordinates to absolute pixel coordinates
    landmark_x = landmark_x_norm * width
    landmark_y = landmark_y_norm * height

    # Draw circles around landmarks for left hand (even indices)
    draw_circles(frame_rgb, landmark_x[::2], landmark_y[::2], radius, (255, 0, 0))  # Red for left hand

    # Draw circles around landmarks for right hand (odd indices)
    draw_circles(frame_rgb, landmark_x[1::2], landmark_y[1::2], radius, (0, 0, 255))  # Blue for right hand

    # Calculate time in seconds
    time_sec = frame_number / fps

    # Plot the frame
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title(f"After cooling frame {frame_number} (time: {time_sec:.2f} s)")

    plt.axis("off")

    # Save the frame as an image in the output directory
    ROI_output_path = os.path.join(output_folder, f"2. After cooling_frame {frame_number}.jpg")
    plt.savefig(ROI_output_path)

    # Close the plot to prevent memory leaks
    plt.close()

# Release the after cooling video capture object
cap.release()

# Determine frame numbers for the first frame, last frame, and in-between frames for the before cooling video
first_frame_number_before = 0
last_frame_number_before = before_length - 1  # 'before_length' is the total number of frames in the before cooling video
in_between_frame_numbers_before = np.linspace(first_frame_number_before, last_frame_number_before, num=5, dtype=int)

# Open the before cooling video file
cap_before = cv2.VideoCapture(before_cooling_video_path)

# Check if the before cooling video file opened successfully
if not cap_before.isOpened():
    print("Error: Unable to open before cooling video file")
    exit()

# Loop over the frame numbers and draw landmarks on the frames for the before cooling video
for frame_number in [first_frame_number_before] + list(in_between_frame_numbers_before) + [last_frame_number_before]:
    # Set the frame position to the desired frame for the before cooling video
    cap_before.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap_before.read()
    if not success:
        print(f"Error: Unable to read frame {frame_number}")
        continue

    # Rotate the frame 90 degrees clockwise
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert the BGR image to RGB for plotting with matplotlib
    frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_BGR2RGB)

    # Define the landmark coordinates for the frame for the left hand
    landmark_x_norm = before_ROIs_x_filtered[:, frame_number]
    landmark_y_norm = before_ROIs_y_filtered[:, frame_number]

    # Get the height and width of the frame
    height, width, _ = frame_rgb.shape

    # Convert normalized coordinates to absolute pixel coordinates
    landmark_x = landmark_x_norm * width
    landmark_y = landmark_y_norm * height

    # Draw circles around landmarks for left hand (even indices)
    draw_circles(frame_rgb, landmark_x[::2], landmark_y[::2], radius, (255, 0, 0))  # Red for left hand

    # Draw circles around landmarks for right hand (odd indices)
    draw_circles(frame_rgb, landmark_x[1::2], landmark_y[1::2], radius, (0, 0, 255))  # Blue for right hand

    # Calculate time in seconds
    time_sec = frame_number / fps

    # Plot the frame
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Before cooling frame {frame_number} (time: {time_sec:.2f} s)")

    plt.axis("off")

    # Save the frame as an image in the output directory
    ROI_output_path = os.path.join(output_folder, f"1. Before cooling_frame {frame_number}.jpg")
    plt.savefig(ROI_output_path)

    # Close the plot to prevent memory leaks
    plt.close()

# Release the before cooling video capture object
cap_before.release()

#%% Determine median temperature in ROIs

# Initialize lists to store minv, maxv, and timestamps for both "after cooling" and "before cooling"
after_minv_values = []
after_maxv_values = []
before_minv_values = []
before_maxv_values = []
after_timestamps = []  # Initialize list for after_cooling_video timestamps
before_timestamps = []  # Initialize list for before_cooling_video timestamps

# Initialize lists to store temperature values for each video type
after_median_temps_list = []
after_std_dev_temps_list = []
before_median_temps_list = []
before_std_dev_temps_list = []

# Define video paths and corresponding ROI coordinates
video_paths = [after_cooling_video_path, before_cooling_video_path]
roi_x_coords = [after_ROIs_x_filtered, before_ROIs_x_filtered]
roi_y_coords = [after_ROIs_y_filtered, before_ROIs_y_filtered]

# Function to get points within a circle
#def get_points_within_circle(center_x, center_y, radius, index):
    # Your implementation of getting points within a circle
    #pass

# Initialize lists to store temperature values
#before_mean_temps = []
#before_std_dev_temps = []

def get_points_within_circle(center_x, center_y, radius, index):
    points = []
    for i in range(-radius*2, radius*2 + 1):
        for j in range(-radius*2, radius*2 + 1):
            x = center_x + i
            y = center_y + j
            if index in [10, 11, 12, 13]:
                if math.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius * 2:
                    points.append((x, y))
            else:
                if math.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius:
                    points.append((x, y))
    return points

# Define function to calculate median temperature values
def median_temp(neighbor_pixel_temps):
    return np.median(neighbor_pixel_temps)

# Define function to calculate absolute deviation from the median
def absolute_deviation_from_median(data, median):
    return np.abs(data - median)

# Define function to calculate standard deviation from the median
def std_from_median(data, median):
    abs_deviations = absolute_deviation_from_median(data, median)
    return np.mean(abs_deviations)

# Initialize counters for out-of-bounds frames
nan_count_out_of_bounds_before = 0
nan_count_out_of_bounds_after = 0

# Loop over video paths
for video_path, roi_x, roi_y in zip(video_paths, roi_x_coords, roi_y_coords):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = len(roi_x[0])
    step_size = 10  # Set the step size to 10 frames

    # Loop over the frames with step size
    for frame_number in range(0, total_frames, step_size):
        # Set the frame position based on the frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        success, frame = cap.read()
        if not success:
            break  # Break the loop if there are no more frames

        # Rotate the frame 90 degrees clockwise
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Extract regions containing min and max temperature values
        if width == 1440:
            minv_frame = frame_rotated[1008:1050, 72:216]
        else:
            minv_frame = frame_rotated[1029:1068, 72:216]

        maxv_frame = frame_rotated[39:81, 72:216]

        # Perform OCR to read min and max temperature values
        custom_config = r'--oem 3 --psm 7'
        minv = pytesseract.image_to_string(minv_frame, config=custom_config)
        maxv = pytesseract.image_to_string(maxv_frame, config=custom_config)

        # Convert temperature values to float
        try:
            minv = float(minv.replace('°', '').replace('C', '').replace(',', '.'))
            minv = math.floor(minv * 10) / 10
        except ValueError:
            minv = np.nan

        try:
            maxv = float(maxv.replace('°', '').replace('C', '').replace(',', '.'))
            maxv = math.floor(maxv * 10) / 10
        except ValueError:
            maxv = np.nan

        # Check for out-of-range values and set them to NaN
        if minv > 30 or maxv > 100:
            print('minv or maxv values are out of range in frame', frame_number, 'so the begin temperature could not be determined')
            minv = np.nan
            maxv = np.nan

        # Append minv and maxv to the respective lists based on video type
        if video_path == after_cooling_video_path:
            after_minv_values.append(minv)
            after_maxv_values.append(maxv)
            after_timestamps.append(frame_number * (1 / cap.get(cv2.CAP_PROP_FPS)))  # Append after_cooling_video timestamp
        else:
            before_minv_values.append(minv)
            before_maxv_values.append(maxv)
            before_timestamps.append(frame_number * (1 / cap.get(cv2.CAP_PROP_FPS)))  # Append before_cooling_video timestamp

        # Perform calculations for each click coordinate
        median_temps_image = []
        std_dev_temps_image = []

        for i in range(len(roi_x)):
            # Check if x or y coordinate is NaN
            if np.isnan(roi_x[i, frame_number]) or np.isnan(roi_y[i, frame_number]):
                # Handle NaN values
                median_temps_image.append(np.nan)
                std_dev_temps_image.append(np.nan)
                continue  # Skip the rest of the loop for this iteration

            x = int(roi_x[i, frame_number] * frame_rotated.shape[1])
            y = int(roi_y[i, frame_number] * frame_rotated.shape[0])

            if x < 0 or x >= frame_rotated.shape[1] or y < 0 or y >= frame_rotated.shape[0]:
                # Coordinate is out of bounds, assign NaN
                median_temps_image.append(np.nan)
                std_dev_temps_image.append(np.nan)
                if video_path == after_cooling_video_path:
                    nan_count_out_of_bounds_after += 1
                else:
                    nan_count_out_of_bounds_before += 1
                continue

            if np.isnan(minv) or np.isnan(maxv):
                median_temp_val = np.nan
                std_dev_temp_val = np.nan
            else:
                points_within_circle = get_points_within_circle(x, y, radius, i)
                # Check if any point within the circle is out of bounds
                points_within_bounds = [(point[0], point[1]) for point in points_within_circle
                                        if point[0] >= 0 and point[0] < frame_rotated.shape[1]
                                        and point[1] >= 0 and point[1] < frame_rotated.shape[0]]
                if len(points_within_bounds) < len(points_within_circle):
                    # At least one point within the circle is out of bounds, assign NaN
                    median_temps_image.append(np.nan)
                    std_dev_temps_image.append(np.nan)
                    if video_path == after_cooling_video_path:
                        nan_count_out_of_bounds_after += 1
                    else:
                        nan_count_out_of_bounds_before += 1
                    continue

                neighbor_pixel_values = [frame_rotated[point[1], point[0]] for point in points_within_circle]
                neighbor_pixel_temps = [(value / 255) * (maxv - minv) + minv for value in neighbor_pixel_values]
                median_temp_val = median_temp(neighbor_pixel_temps)
                std_dev_temp_val = std_from_median(neighbor_pixel_temps, median_temp_val)

            median_temps_image.append(median_temp_val)
            std_dev_temps_image.append(std_dev_temp_val)

        # Store the results for this image based on video type
        if video_path == after_cooling_video_path:
            after_median_temps_list.append(median_temps_image)
            after_std_dev_temps_list.append(std_dev_temps_image)
        else:
            before_median_temps_list.append(median_temps_image)
            before_std_dev_temps_list.append(std_dev_temps_image)

    # Print the total count of frames with out-of-bounds coordinates
    # print("Total number of frames with out-of-bounds coordinates:", nan_count_out_of_bounds)

    # Release the video capture object
    cap.release()

# Convert the lists of results to NumPy arrays
after_minv_values = np.array(after_minv_values)
after_maxv_values = np.array(after_maxv_values)
before_minv_values = np.array(before_minv_values)
before_maxv_values = np.array(before_maxv_values)
after_timestamps = np.array(after_timestamps)  # Convert to NumPy array
before_timestamps = np.array(before_timestamps)  # Convert to NumPy array
after_median_temps = np.array(after_median_temps_list)
after_std_dev_temps = np.array(after_std_dev_temps_list)
before_median_temps = np.array(before_median_temps_list)
before_std_dev_temps = np.array(before_std_dev_temps_list)

# Check for NaN values in temperature values and count them
nan_count_after_minv = np.sum(np.isnan(after_minv_values))
nan_count_after_maxv = np.sum(np.isnan(after_maxv_values))
nan_count_after_temps = np.sum(np.isnan(after_median_temps).any(axis=1))

nan_count_before_minv = np.sum(np.isnan(before_minv_values))
nan_count_before_maxv = np.sum(np.isnan(before_maxv_values))
nan_count_before_temps = np.sum(np.isnan(before_median_temps).any(axis=1))

#%% Plot 1: Temperature development of the 12 ROIs

# Calculate the median temperature for each landmark across all frames
before_median_temps_avg = np.nanmedian(before_median_temps, axis=0)
before_std_dev_temps_avg = np.nanmedian(before_std_dev_temps, axis=0)

# Convert the NumPy array to a list
before_median_temps_avg = before_median_temps_avg.tolist()
before_std_dev_temps_avg = before_std_dev_temps_avg.tolist()

# Define columns to plot
columns_to_plot_median = []

# Check for columns with all zeros and add non-zero columns to the lists
for i in range(after_median_temps.shape[1]):
    if not np.all(after_median_temps[:, i] == 0):  # Check only median temperatures for non-zero columns
        columns_to_plot_median.append(after_median_temps[:, i])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figsize as needed

# Plot the selected columns
for index, median_column in enumerate(columns_to_plot_median):
    linestyle = '-' if index % 2 == 0 else '--'  # Adjusted linestyle
    ax.plot(np.array(after_timestamps) / 1, median_column, label=labels[index], linestyle=linestyle, color=custom_colors[index], linewidth=1.5)  # Set linewidth to 1.5

# Add labels and title
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Temperature [C]')
ax.set_title('Temperature development for the 12 ROIs', fontweight='bold')

# Create custom legend
handles, legend_labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=4)

# Adjust the linestyle for the legend lines
for i, line in enumerate(legend.get_lines()):
    line.set_linestyle('-' if i % 2 == 0 else '--')  # Ensure the legend reflects the correct linestyle

# Show the plot
plot_filename = os.path.join(output_path, f"E1. Temperature plot {video_name}.png")
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()

#%% Plot 2: Temperature development per hand

# Define columns to plot for left and right hands
columns_to_plot_median_L = []
columns_to_plot_median_R = []

# Check for columns with all zeros and add non-zero columns to the lists
for i in range(after_median_temps.shape[1]):
    if i % 2 == 0:
        if not np.all(after_median_temps[:, i] == 0):
            columns_to_plot_median_L.append(after_median_temps[:, i])
    else:
        if not np.all(after_median_temps[:, i] == 0):
            columns_to_plot_median_R.append(after_median_temps[:, i])

# Create a figure and axes for the three plots
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

# Create subplots within the gridspec
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1, sharey=ax1)  # Shares X and Y axis with ax1
ax3 = plt.subplot(gs[2], sharex=ax1)

# Plot the selected columns for the left hand ("L" labels) with normal lines
for index, median_column in enumerate(columns_to_plot_median_L):
    ax1.plot(after_timestamps, median_column, label=labels[index * 2], linestyle='-', color=custom_colors[index * 2], linewidth=1.5)

# Add labels and legend for the first plot
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Temperature [C]')
ax1.set_title('Left hand')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)

# Plot the selected columns for the right hand ("R" labels) with dotted lines
for index, median_column in enumerate(columns_to_plot_median_R):
    ax2.plot(after_timestamps, median_column, label=labels[index * 2 + 1], linestyle='--', color=custom_colors[index * 2 + 1], linewidth=1.5)

# Add labels and legend for the second plot
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Temperature [C]')
ax2.set_title('Right hand')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)

# Calculate and plot the difference between the right hand and left hand temperatures using second color of each pair
for index, (median_L, median_R) in enumerate(zip(columns_to_plot_median_L, columns_to_plot_median_R)):
    difference = np.array(median_R) - np.array(median_L)
    ax3.plot(after_timestamps, difference, label=f'{labels[index * 2 + 1]} - {labels[index * 2]}', linestyle='-', color=custom_colors[index * 2 + 1], linewidth=1.5)

# Add labels and legend for the third plot
ax3.set_xlabel('Time [sec]')
ax3.set_ylabel('Temperature difference [C]')
ax3.set_title('Temperature difference between right and left hand')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)

# Add general title for all subplots
fig.suptitle('Temperature development per hand', y=0.98, fontsize=16, fontweight='bold')

# Save and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for the title
plot_filename = os.path.join(output_path, f"E3. Temperature plot per hand {video_name}.png")
plt.savefig(plot_filename)
plt.show()

#%% Moving average

# Function to calculate the moving average for a single column
def moving_average(data, initial_window_size):
    smoothed_data = np.copy(data)
    
    for i in range(len(data)):
        current_window_size = initial_window_size
        while True:
            half_window = current_window_size // 2
            window_start = max(0, i - half_window)
            window_end = min(len(data), i + half_window + 1)
            window_values = data[window_start:window_end]
            non_nan_values = window_values[~np.isnan(window_values)]

            if len(non_nan_values) > 0:
                smoothed_data[i] = np.mean(non_nan_values)
                break
            elif current_window_size >= len(data):
                smoothed_data[i] = np.nan
                break
            else:
                current_window_size += 5  # Increase the window size by 5

    return smoothed_data

# Function to apply the moving average to each column of a 2D array
def moving_average_per_column(data, initial_window_size):
    smoothed_data = np.copy(data)

    for i in range(data.shape[1]):
        col_data = data[:, i]
        if np.all(np.isnan(col_data)):
            smoothed_data[:, i] = col_data  # Keep NaNs if the whole column is NaNs
        else:
            smoothed_data[:, i] = moving_average(col_data, initial_window_size)

    return smoothed_data

window_size_after = 50
window_size_before = 5

# Apply to the temperature data after the cooling process
after_median_smoothed = moving_average_per_column(after_median_temps, window_size_after)

# Apply to the temperature data before the cooling process
before_median_smoothed = moving_average_per_column(before_median_temps, window_size_before)

#%% Plot 1 - Smoothed

# Calculate the median temperature for each landmark across all frames
before_median_smoothed_avg = np.nanmedian(before_median_smoothed, axis=0)
# before_std_dev_smoothed_avg = np.nanmedian(before_std_dev_smoothed, axis=0)

# Convert the NumPy array to a list
before_median_smoothed_avg = before_median_smoothed_avg.tolist()
#before_std_dev_temps_avg = before_std_dev_temps_avg.tolist()

# Define columns to plot
columns_to_plot_median_smoothed = []

# Check for columns with all zeros and add non-zero columns to the lists
for i in range(after_median_smoothed.shape[1]):
    if not np.all(after_median_smoothed[:, i] == 0):  # Check only smoothed median temperatures for non-zero columns
        columns_to_plot_median_smoothed.append(after_median_smoothed[:, i])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figsize as needed

# Plot the selected columns
for index, median_column in enumerate(columns_to_plot_median_smoothed):
    linestyle = '-' if index % 2 == 0 else '--'  # Adjusted linestyle
    ax.plot(np.array(after_timestamps) / 1, median_column, label=labels[index], linestyle=linestyle, color=custom_colors[index], linewidth=1.5)  # Set linewidth to 1.5

# Add labels and title
ax.set_xlabel('Time [sec]')
ax.set_ylabel('Temperature [C]')
ax.set_title('Temperature development for the 12 ROIs - smoothed', fontweight='bold')

# Create custom legend
handles, legend_labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=4)

# Adjust the linestyle for the legend lines
for i, line in enumerate(legend.get_lines()):
    line.set_linestyle('-' if i % 2 == 0 else '--')  # Ensure the legend reflects the correct linestyle

# Show the plot
plot_filename = os.path.join(output_path, f"E2. Temperature plot {video_name} - smoothed.png")
plt.savefig(plot_filename, bbox_inches='tight')
plt.show()

#%% Plot 2 - Smoothed

# Define columns to plot for left and right hands
columns_to_plot_median_L = []
columns_to_plot_median_R = []

# Check for columns with all zeros and add non-zero columns to the lists
for i in range(after_median_smoothed.shape[1]):
    if i % 2 == 0:
        if not np.all(after_median_smoothed[:, i] == 0):
            columns_to_plot_median_L.append(after_median_smoothed[:, i])
    else:
        if not np.all(after_median_smoothed[:, i] == 0):
            columns_to_plot_median_R.append(after_median_smoothed[:, i])

# Create a figure and axes for the three plots
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

# Create subplots within the gridspec
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1, sharey=ax1)  # Shares X and Y axis with ax1
ax3 = plt.subplot(gs[2], sharex=ax1)

# Plot the selected columns for the left hand ("L" labels) with normal lines
for index, median_column in enumerate(columns_to_plot_median_L):
    ax1.plot(after_timestamps, median_column, label=labels[index * 2], linestyle='-', color=custom_colors[index * 2], linewidth=1.5)

# Add labels and legend for the first plot
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Temperature [C]')
ax1.set_title('Left hand')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)

# Plot the selected columns for the right hand ("R" labels) with dotted lines
for index, median_column in enumerate(columns_to_plot_median_R):
    ax2.plot(after_timestamps, median_column, label=labels[index * 2 + 1], linestyle='--', color=custom_colors[index * 2 + 1], linewidth=1.5)

# Add labels and legend for the second plot
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Temperature [C]')
ax2.set_title('Right hand')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)

# Calculate and plot the difference between the right hand and left hand temperatures using second color of each pair
for index, (median_L, median_R) in enumerate(zip(columns_to_plot_median_L, columns_to_plot_median_R)):
    difference = np.array(median_R) - np.array(median_L)
    ax3.plot(after_timestamps, difference, label=f'{labels[index * 2 + 1]} - {labels[index * 2]}', linestyle='-', color=custom_colors[index * 2 + 1], linewidth=1.5)

# Add labels and legend for the third plot
ax3.set_xlabel('Time [sec]')
ax3.set_ylabel('Temperature difference [C]')
ax3.set_title('Temperature difference between right and left hand')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)

# Add general title for all subplots
fig.suptitle('Temperature development per hand', y=0.98, fontsize=16, fontweight='bold')

# Save and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for the title
plot_filename = os.path.join(output_path, f"E4. Temperature plot per hand {video_name} - smoothed.png")
plt.savefig(plot_filename)
plt.show()

#%% Delete before and after cooling videos from output folder

# Delete the before cooling video file to save memory
if os.path.exists(before_cooling_video_path):
    os.remove(before_cooling_video_path)

# Delete the after cooling video file to save memory
if os.path.exists(after_cooling_video_path):
    os.remove(after_cooling_video_path)
    
#%% Move video to processed videos folder (not for repeatibility test)

# Move the video to the 'Processed videos' folder
processed_videos_folder = os.path.join(parent_directory, '2. Processed videos')
shutil.move(file_path, processed_videos_folder)

#%% Run time calculation

# Record the end time
end_time = time.time()

# Calculate the total runtime
total_time = end_time - start_time

#%% Variables of interest

temp_base = before_median_smoothed_avg

# Define the time points for extraction
time_points = [0, 5*60, 10*60, 15*60]  # in seconds

# Function to find the closest index
def find_closest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Extract the indexes for the defined time points
indexes = [find_closest_index(after_timestamps, t) for t in time_points]

# Extract the values for each time point
temp_cpt = after_median_smoothed[indexes[0], :]  # Values at t = 0
temp_5 = after_median_smoothed[indexes[1], :]  # Values at t = 300 seconds (5 minutes)
temp_10 = after_median_smoothed[indexes[2], :]  # Values at t = 600 seconds (10 minutes)
temp_15 = after_median_smoothed[indexes[3], :]  # Values at t = 900 seconds (15 minutes)

temp_drop = temp_base - temp_cpt

# Calculate recovery as a percentage of temp_drop 
recovery_5 = (temp_5 - temp_cpt) / temp_drop * 100 
recovery_10 = (temp_10 - temp_cpt) / temp_drop * 100 
recovery_15 = (temp_15 - temp_cpt) / temp_drop * 100

# Calculate recovery thresholds
threshold_50 = temp_drop * 0.5
threshold_95 = temp_drop * 0.95

# Initialize lists to store t_50 and t_95 for each ROI
t_50 = []
t_95 = []

# Calculate the time till 50% and 95% recovery for each ROI
for roi in range(12):
    # Find the index where the recovery exceeds 50%
    t_50_idx = np.argmax(after_median_smoothed[:, roi] >= (temp_cpt[roi] + threshold_50[roi]))
    t_50_value = after_timestamps[t_50_idx] if after_median_smoothed[t_50_idx, roi] >= (temp_cpt[roi] + threshold_50[roi]) else np.nan
    t_50.append(t_50_value / 60.0)  # Convert to minutes
    
    # Find the index where the recovery exceeds 95%
    t_95_idx = np.argmax(after_median_smoothed[:, roi] >= (temp_cpt[roi] + threshold_95[roi]))
    t_95_value = after_timestamps[t_95_idx] if after_median_smoothed[t_95_idx, roi] >= (temp_cpt[roi] + threshold_95[roi]) else np.nan
    t_95.append(t_95_value / 60.0)  # Convert to minutes
    
# Ensure all values are numpy arrays
temp_base = np.array(temp_base)
temp_cpt = np.array(temp_cpt)
temp_5 = np.array(temp_5)
temp_10 = np.array(temp_10)
temp_15 = np.array(temp_15)
temp_drop = np.array(temp_drop)
recovery_5 = np.array(recovery_5)
recovery_10 = np.array(recovery_10)
recovery_15 = np.array(recovery_15)
t_50 = np.array(t_50)
t_95 = np.array(t_95)

#%% Performance report (nog checken of percentages kloppen)

# Calculate the percentage of coordinates that could not be tracked
percent_nan_before_left = round((num_nan_before_left / num_frames_before) * 100, 1)
percent_nan_before_right = round((num_nan_before_right / num_frames_before) * 100, 1)
percent_nan_after_left = round((num_nan_after_left / num_frames_after) * 100, 1)
percent_nan_after_right = round((num_nan_after_right / num_frames_after) * 100, 1)

# Calculate the percentages of the temperature values that could not be determined
# Calculate the total number of values for before and after cooling phases
total_before_values = before_minv_values.size
total_before_temps = before_median_temps.shape[0]
total_after_values = after_minv_values.size
total_after_temps = after_median_temps.shape[0]

# Calculate percentages of NaN values for before cooling phase
percent_nan_before_minv = round((nan_count_before_minv / total_before_values) * 100, 1)
percent_nan_before_maxv = round((nan_count_before_maxv / total_before_values) * 100, 1)
percent_nan_before_outofbounds = round((nan_count_out_of_bounds_before / total_before_values) * 100, 1)
percent_nan_before_temps = round((nan_count_before_temps / total_before_temps) * 100, 1)

# Calculate percentages of NaN values for after cooling phase
percent_nan_after_minv = round((nan_count_after_minv / total_after_values) * 100, 1)
percent_nan_after_maxv = round((nan_count_after_maxv / total_after_values) * 100, 1)
percent_nan_after_outofbounds = round((nan_count_out_of_bounds_after / total_after_values) * 100, 1)
percent_nan_after_temps = round((nan_count_after_temps / total_after_temps) * 100, 1)

# Calculate the total runtime in minutes and round to 0 decimals
total_time_minutes = round(total_time / 60, 0)

# Quantitative analysis of filtering
def calculate_euclidean_distances(x_unfiltered, y_unfiltered, x_filtered, y_filtered):
    return np.sqrt((x_unfiltered - x_filtered) ** 2 + (y_unfiltered - y_filtered) ** 2)

def calculate_combined_statistics(euclidean_distances):
    mean_euc_dist = round(np.nanmean(euclidean_distances), 2)
    max_euc_dist = round(np.nanmax(euclidean_distances), 2)
    std_euc_dist = round(np.nanstd(euclidean_distances), 2)
    return mean_euc_dist, max_euc_dist, std_euc_dist

# Calculate Euclidean distances for after cooling
euc_dist_after = calculate_euclidean_distances(after_ROIs_x, after_ROIs_y, after_ROIs_x_filtered, after_ROIs_y_filtered)

# Calculate Euclidean distances for before cooling
euc_dist_before = calculate_euclidean_distances(before_ROIs_x, before_ROIs_y, before_ROIs_x_filtered, before_ROIs_y_filtered)

# Calculate combined statistics for Euclidean distances across all ROIs
mean_euc_dist_after, max_euc_dist_after, std_euc_dist_after = calculate_combined_statistics(euc_dist_after)
mean_euc_dist_before, max_euc_dist_before, std_euc_dist_before = calculate_combined_statistics(euc_dist_before)

       
#%% Export results to Excel

# Define variables and descriptions
variables = ["Temp_Base [C]", "Temp_CPT [C]", "Temp_5 [C]", "Temp_10 [C]", "Temp_15 [C]", "Temp_Drop [C]", "Recovery_5 [C]", "Recovery_10 [C]", "Recovery_15 [C]", "t_50 [min]", "t_95 [min]"]
values = [temp_base, temp_cpt, temp_5, temp_10, temp_15, temp_drop, recovery_5, recovery_10, recovery_15, t_50, t_95]
descriptions = [
    "The baseline skin temperature in a ROI of the participant before applying CPT",
    "The skin temperature in a ROI of the participant directly after CPT",
    "Temperature in a ROI at 5 minutes time since removing CPT. Derived from Temp_i",
    "Temperature in a ROI at 10 minutes time since removing CPT. Derived from Temp_i",
    "Temperature in a ROI at 15 minutes time since removing CPT. Derived from Temp_i",
    "Temperature drop in a ROI caused by CPT, calculated from Baseline temp - Temp CPT",
    "Recovery at 5 minutes since removing CPT, calculated as a percentage of temperature drop. Derived from Recovery_i",
    "Recovery at 10 minutes since removing CPT, calculated as a percentage of temperature drop. Derived from Recovery_i",
    "Recovery at 15 minutes since removing CPT, calculated as a percentage of temperature drop. Derived from Recovery_i",
    "Duration to reach 50% recovery of temperature drop after CPT. Derived from Recovery_i",
    "Duration to reach 95% recovery of temperature drop after CPT. Derived from Recovery_i"
]

# Create a DataFrame for the variables of interest
variables_of_interest_df = pd.DataFrame({
    "Variable": variables,
    "Description": descriptions,
})

# Add columns for each label
for i, label in enumerate(labels):
    variables_of_interest_df[label] = [np.array2string(value[i], precision=2, separator=', ') for value in values]

# Calculate the average of the ROIs and add it to the DataFrame
variables_of_interest_df["Average"] = variables_of_interest_df[labels].astype(float).mean(axis=1).round(2)  # <-- Added

# Specify the name and path for the Excel file
output_file = os.path.join(output_path, f"Results {video_name}.xlsx")

model_performance_data = {
    "Measure": [
        "Before cooling frames where left hand coordinates could not be tracked",
        "Before cooling frames where right hand coordinates could not be tracked",
        "",
        "After cooling frames where left hand coordinates could not be tracked",
        "After cooling frames where right hand coordinates could not be tracked",
        "",
        "Before cooling mean euclidean distance between unfiltered and filtered ROI coordinates",
        "Before cooling max euclidean distance between unfiltered and filtered ROI coordinates",
        "Before cooling euclidean distance standard deviations between unfiltered and filtered ROI coordinates",
        "",
        "After cooling mean euclidean distance between unfiltered and filtered ROI coordinates",
        "After cooling max euclidean distance between unfiltered and filtered ROI coordinates",
        "After cooling euclidean distance standard deviations between unfiltered and filtered ROI coordinates",
        "",
        "Before cooling frames where minimum temperature could not be read from the scale",
        "Before cooling frames where maximum temperature could not be read from the scale",
        "Before cooling frames where ROI (radius 10 around coordinates) exceed the frame boundaries",
        "Before cooling frames where ROI temperature could not be determined",
        "",
        "After cooling frames where minimum temperature could not be read from the scale",
        "After cooling frames where maximum temperature could not be read from the scale",
        "After cooling frames where ROI (radius 10 around coordinates) exceed the frame boundaries",
        "After cooling frames where ROI temperature could not be determined",
        
        "",
        "",
        "Model running time [minutes]"
    ],
    "Value [#]": [
        num_nan_before_left,
        num_nan_before_right,
        "",
        num_nan_after_left,
        num_nan_after_right,
        "",
        mean_euc_dist_before,
        max_euc_dist_before,
        std_euc_dist_before,
        "",
        mean_euc_dist_after,
        max_euc_dist_after,
        std_euc_dist_after,
        "",
        nan_count_before_minv,
        nan_count_before_maxv,
        nan_count_out_of_bounds_before,
        nan_count_before_temps,
        "",
        nan_count_after_minv,
        nan_count_after_maxv,
        nan_count_out_of_bounds_after,
        nan_count_after_temps,
        
        "",
        "",
        total_time_minutes
    ],
    "Percentage [%]": [
        percent_nan_before_left,
        percent_nan_before_right,
        "",
        percent_nan_after_left,
        percent_nan_after_right,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        percent_nan_before_minv,
        percent_nan_before_maxv,
        percent_nan_before_outofbounds,
        percent_nan_before_temps,
        "",
        percent_nan_after_minv,
        percent_nan_after_maxv,
        percent_nan_after_outofbounds,
        percent_nan_after_temps,
        
        "",
        "",
        ""
    ]
}

# Create DataFrame for temperature values
df1 = pd.DataFrame(after_median_smoothed)
df1.columns = labels
new_df1 = pd.DataFrame([before_median_smoothed_avg], columns=df1.columns)
df1 = pd.concat([new_df1, df1], ignore_index=True)

# Round the time values to 2 decimals
rounded_timestamps = np.round(after_timestamps, 2)

# Adjust the length of rounded_timestamps
if len(rounded_timestamps) == len(df1) - 1:
    rounded_timestamps = np.append("Before cooling", rounded_timestamps)
elif len(rounded_timestamps) == len(df1) + 1:
    rounded_timestamps = rounded_timestamps[1:]

# Include the time column in the DataFrame
df1.insert(0, "Time [sec]", rounded_timestamps)

# Replace "." with "," in the variables of interest DataFrame
#variables_of_interest_df = variables_of_interest_df.applymap(lambda x: x.replace('.', ',') if isinstance(x, str) else x)  #Werkt wel maar excel geeft melding dat het dan tekst is ipv een waarde

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Create a worksheet for variables of interest
    variables_of_interest_df.to_excel(writer, sheet_name='Variables of interest', index=False)

    # Create a worksheet for temperature values
    df1.round(2).to_excel(writer, sheet_name='Temperature values', index=False)

    # Create a worksheet for model performance
    model_performance_df = pd.DataFrame(model_performance_data)
    model_performance_df.round(2).to_excel(writer, sheet_name='Model performance', index=False)

    # Add headers for time column
    bold_format = writer.book.add_format({'bold': True})
    sheet = writer.sheets['Temperature values']
    sheet.write(0, 0, 'Time [sec]', bold_format)

    # Freeze the top row and left column
    sheet.freeze_panes(1, 1)
    
    # Adjust cell width to fit content for Temperature values sheet
    for i, col in enumerate(df1.columns):
        column_len = max(df1[col].astype(str).str.len().max(), len(col)) + 2
        sheet.set_column(i, i, column_len)

    # Adjust cell width to fit content for Model performance sheet
    model_performance_sheet = writer.sheets['Model performance']
    for i, col in enumerate(model_performance_df.columns):
        column_len = max(model_performance_df[col].astype(str).str.len().max(), len(col)) + 2
        model_performance_sheet.set_column(i, i, column_len)

    # Adjust cell width to fit content for Variables of interest sheet
    variables_of_interest_sheet = writer.sheets['Variables of interest']
    for i, col in enumerate(variables_of_interest_df.columns):
        column_len = max(variables_of_interest_df[col].astype(str).str.len().max(), len(col)) + 2
        variables_of_interest_sheet.set_column(i, i, column_len)
        