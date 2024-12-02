import cv2
import os
import xml.etree.ElementTree as ET
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import numpy as np
from datetime import datetime

def parse_opencv_matrix(matrix_elem):
    """Parse OpenCV matrix data from XML element."""
    data = matrix_elem.find('data').text.strip().split()
    return [float(x) for x in data]

def convert_leapmotion_to_mediapipe(x, y, z, image_width, image_height):
    """Convert Leap Motion coordinates to MediaPipe format and pixel coordinates"""
    # Get bounding box info from XML
    x_min, x_max = 306, 352  # From XML
    y_min, y_max = 248, 356  # From XML
    
    # Center offset based on palm position
    x_palm, y_palm = 323, 299  # From XML
    
    # Convert to image coordinates first
    image_x = int(x_palm + (x * (x_max - x_min) / 200.0))  # Scale x by box width
    image_y = int(y_palm - (y - 200) * (y_max - y_min) / 100.0)  # Offset y by 200mm baseline
    
    # Normalize to [0,1] range for MediaPipe
    normalized_x = float(image_x) / image_width
    normalized_y = float(image_y) / image_height
    normalized_z = z / 100.0  # Normalize z to roughly [0,1] range
    
    # Debug info
    debug_info = f"\nOriginal: ({x:.2f}, {y:.2f}, {z:.2f})"
    debug_info += f"\nImage coords: ({image_x}, {image_y})"
    debug_info += f"\nNormalized: ({normalized_x:.3f}, {normalized_y:.3f}, {normalized_z:.3f})"
    
    return (image_x, image_y), (normalized_x, normalized_y, normalized_z), debug_info

def visualize_gt_landmarks():
    # Setup debug log file
    debug_dir = "debug_logs"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = os.path.join(debug_dir, f"debug_log_{timestamp}.txt")
    
    with open(debug_file, 'w') as f:
        # Load the XML file
        xml_path = "/home/user/Desktop/RP/datasets/multimodalkaggle/skeletal/07/test_gesture/21_right/01/frame_123750.xml"
        f.write(f"\nLoading XML from: {xml_path}\n")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Debug print XML structure
        f.write("\nXML Structure:\n")
        for elem in root.iter():
            f.write(f"Tag: {elem.tag}\n")
            if elem.text and elem.text.strip():
                f.write(f"  Text: {elem.text.strip()[:50]}...\n")
        
        # Load corresponding image with absolute path
        image_path = "/home/user/Desktop/RP/datasets/multimodalkaggle/near-infrared/07/test_gesture/21_right/01/frame_123750_r.png"
        f.write(f"\nLoading image from: {image_path}\n")
        
        image = cv2.imread(image_path)
        if image is None:
            error_msg = f"Could not read image at {image_path}"
            f.write(f"ERROR: {error_msg}\n")
            raise ValueError(error_msg)
        
        # Debug image dimensions
        height, width = image.shape[:2]
        f.write(f"\nImage dimensions: {width}x{height}\n")
        
        # Create debug image
        debug_image = image.copy()
        
        # Print full XML content for debugging
        f.write("\nFull XML content:\n")
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        f.write(xml_str)
        
        # Debug both RigthImage and LeftImage coordinates
        right_image = root.find('.//RigthImage')
        left_image = root.find('.//LeftImage')
        
        f.write("\nRigthImage coordinates:\n")
        if right_image is not None:
            f.write(f"x_min: {right_image.find('x_min').text}\n")
            f.write(f"x_max: {right_image.find('x_max').text}\n")
            f.write(f"y_min: {right_image.find('y_min').text}\n")
            f.write(f"y_max: {right_image.find('y_max').text}\n")
            f.write(f"x_palm: {right_image.find('x_palm').text}\n")
            f.write(f"y_palm: {right_image.find('y_palm').text}\n")
        
        f.write("\nLeftImage coordinates:\n")
        if left_image is not None:
            f.write(f"x_min: {left_image.find('x_min').text}\n")
            f.write(f"x_max: {left_image.find('x_max').text}\n")
            f.write(f"y_min: {left_image.find('y_min').text}\n")
            f.write(f"y_max: {left_image.find('y_max').text}\n")
            f.write(f"x_palm: {left_image.find('x_palm').text}\n")
            f.write(f"y_palm: {left_image.find('y_palm').text}\n")
        
        # Draw both bounding boxes with different colors
        if right_image is not None:
            x_min = int(right_image.find('x_min').text)
            x_max = int(right_image.find('x_max').text)
            y_min = int(right_image.find('y_min').text)
            y_max = int(right_image.find('y_max').text)
            x_palm = int(right_image.find('x_palm').text)
            y_palm = int(right_image.find('y_palm').text)
            
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for RigthImage
            cv2.circle(debug_image, (x_palm, y_palm), 5, (0, 255, 0), -1)
            cv2.putText(debug_image, f"Right ({x_palm}, {y_palm})", (x_palm+10, y_palm),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            f.write(f"\nDrawn RigthImage box at: ({x_min}, {y_min}) to ({x_max}, {y_max})\n")
            f.write(f"RigthImage palm at: ({x_palm}, {y_palm})\n")
        
        if left_image is not None:
            x_min = int(left_image.find('x_min').text)
            x_max = int(left_image.find('x_max').text)
            y_min = int(left_image.find('y_min').text)
            y_max = int(left_image.find('y_max').text)
            x_palm = int(left_image.find('x_palm').text)
            y_palm = int(left_image.find('y_palm').text)
            
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for LeftImage
            cv2.circle(debug_image, (x_palm, y_palm), 5, (255, 0, 0), -1)
            cv2.putText(debug_image, f"Left ({x_palm}, {y_palm})", (x_palm+10, y_palm),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            f.write(f"\nDrawn LeftImage box at: ({x_min}, {y_min}) to ({x_max}, {y_max})\n")
            f.write(f"LeftImage palm at: ({x_palm}, {y_palm})\n")
        
        # Add debug visualization
        debug_colors = {
            'Thumb': (255, 0, 0),    # Blue
            'Index': (0, 255, 0),    # Green
            'Middle': (0, 255, 255), # Yellow
            'Ring': (0, 165, 255),   # Orange
            'Pinky': (0, 0, 255)     # Red
        }
        
        # Process hand landmarks with enhanced debugging
        right_image = root.find('.//RigthImage')
        if right_image is not None:
            hands = right_image.find('.//Hands')
            if hands is not None:
                right_hand = hands.find('.//Right')
                if right_hand is not None:
                    fingers = right_hand.find('.//Fingers')
                    if fingers is not None:
                        f.write("\n\n=== Hand Landmark Debug Information ===\n")
                        
                        # Get hand bounding box for reference
                        x_min = int(right_image.find('x_min').text)
                        x_max = int(right_image.find('x_max').text)
                        y_min = int(right_image.find('y_min').text)
                        y_max = int(right_image.find('y_max').text)
                        
                        f.write(f"\nHand Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                        
                        for finger in fingers.findall('.//*[TipPosition]'):
                            finger_type = finger.find('Type').text
                            finger_name = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'][int(finger_type)]
                            color = debug_colors[finger_name]
                            
                            f.write(f"\n\n--- {finger_name} Finger ---")
                            
                            # Get positions with debug info
                            tip_pos = parse_opencv_matrix(finger.find('.//TipPosition'))
                            dip_pos = parse_opencv_matrix(finger.find('.//dipPosition'))
                            pip_pos = parse_opencv_matrix(finger.find('.//pipPosition'))
                            mcp_pos = parse_opencv_matrix(finger.find('.//mcpPosition'))
                            
                            # Convert with debug info
                            (tip_px, tip_py), (tip_x, tip_y, tip_z), debug_tip = convert_leapmotion_to_mediapipe(
                                tip_pos[0], tip_pos[1], tip_pos[2], width, height)
                            (dip_px, dip_py), (dip_x, dip_y, dip_z), debug_dip = convert_leapmotion_to_mediapipe(
                                dip_pos[0], dip_pos[1], dip_pos[2], width, height)
                            (pip_px, pip_py), (pip_x, pip_y, pip_z), debug_pip = convert_leapmotion_to_mediapipe(
                                pip_pos[0], pip_pos[1], pip_pos[2], width, height)
                            (mcp_px, mcp_py), (mcp_x, mcp_y, mcp_z), debug_mcp = convert_leapmotion_to_mediapipe(
                                mcp_pos[0], mcp_pos[1], mcp_pos[2], width, height)
                            
                            # Write debug info
                            f.write(f"\nTIP:{debug_tip}")
                            f.write(f"\nDIP:{debug_dip}")
                            f.write(f"\nPIP:{debug_pip}")
                            f.write(f"\nMCP:{debug_mcp}")
                            
                            # Draw landmarks with finger-specific colors using pixel coordinates
                            cv2.circle(debug_image, (tip_px, tip_py), 4, color, -1)
                            cv2.circle(debug_image, (dip_px, dip_py), 4, color, -1)
                            cv2.circle(debug_image, (pip_px, pip_py), 4, color, -1)
                            cv2.circle(debug_image, (mcp_px, mcp_py), 4, color, -1)
                            
                            # Draw connections using pixel coordinates
                            cv2.line(debug_image, (tip_px, tip_py), (dip_px, dip_py), color, 2)
                            cv2.line(debug_image, (dip_px, dip_py), (pip_px, pip_py), color, 2)
                            cv2.line(debug_image, (pip_px, pip_py), (mcp_px, mcp_py), color, 2)
                            
                            # Add landmark labels
                            cv2.putText(debug_image, f"{finger_name}_TIP", (tip_px+5, tip_py), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add debug visualization
        f.write(f"\nBounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        f.write(f"\nPalm Center: ({x_palm}, {y_palm})")
        
        # Draw bounding box
        cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
        # Draw palm center
        cv2.circle(debug_image, (x_palm, y_palm), 5, (255, 255, 255), -1)
        
        # Save debug image
        debug_image_path = os.path.join(debug_dir, f"debug_image_{timestamp}.png")
        cv2.imwrite(debug_image_path, debug_image)
        f.write(f"\nDebug image saved to: {debug_image_path}\n")
        
        # Show debug image
        cv2.imshow('Debug View', debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Debug information has been written to: {debug_file}")

if __name__ == "__main__":
    visualize_gt_landmarks()