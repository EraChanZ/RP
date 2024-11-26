import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

def read_leap_data_et(xml_path):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise IOError(f"Failed to parse XML file: {e}")
    
    hand_data = {}
    sphere_data = {}
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    # Prioritize LeftImage first
    image_types = ['LeftImage', 'RigthImage']
    
    for image_type in image_types:
        hands = root.find(f'Frame/Images/{image_type}/Hands')
        if hands is None:
            print(f"Warning: No Hands found in {image_type}")
            continue
        
        right_hand = hands.find('Right')
        if right_hand is None:
            print(f"Warning: No Right hand found in {image_type}")
            continue
        
        # Extract SphereCenter and SphereRadius
        sphere_center_elem = right_hand.find('SphereCenter/data')
        sphere_radius_elem = right_hand.find('SphereRadius')
        
        if sphere_center_elem is not None and sphere_radius_elem is not None:
            try:
                sphere_center = np.array(list(map(float, sphere_center_elem.text.strip().split())))
                sphere_radius = float(sphere_radius_elem.text.strip())
                sphere_data = {
                    'center': sphere_center,
                    'radius': sphere_radius
                }
                print(f"Extracted Sphere Data from {image_type}:")
                print(f"  Center: {sphere_center}")
                print(f"  Radius: {sphere_radius}")
            except (AttributeError, ValueError) as e:
                print(f"Error processing Sphere data in {image_type}: {e}")
        
        fingers_data = right_hand.find('Fingers')
        if fingers_data is None:
            print(f"Warning: No Fingers data found in {image_type}")
            continue
        
        for finger in fingers:
            # Skip if already processed from LeftImage
            if finger in hand_data:
                continue
                
            finger_elem = fingers_data.find(finger)
            if finger_elem is None:
                print(f"Warning: Missing {finger} data in {image_type}")
                continue
            
            try:
                tip_text = finger_elem.find('TipPosition/data').text.strip()
                dip_text = finger_elem.find('dipPosition/data').text.strip()
                pip_text = finger_elem.find('pipPosition/data').text.strip()
                mcp_text = finger_elem.find('mcpPosition/data').text.strip()
                is_extended_text = finger_elem.find('isExtended').text.strip()
                
                tip = np.array(list(map(float, tip_text.split())))
                dip = np.array(list(map(float, dip_text.split())))
                pip = np.array(list(map(float, pip_text.split())))
                mcp = np.array(list(map(float, mcp_text.split())))
                is_extended = float(is_extended_text)
                
                print(f"Found data for {finger} in {image_type}:")
                print(f"  Tip: {tip}")
                print(f"  DIP: {dip}")
                print(f"  PIP: {pip}")
                print(f"  MCP: {mcp}")
                print(f"  isExtended: {is_extended}")
                
                hand_data[finger] = {
                    'tip': tip,
                    'dip': dip,
                    'pip': pip,
                    'mcp': mcp,
                    'isExtended': is_extended
                }
            except AttributeError as e:
                print(f"Error processing {finger} in {image_type}: {e}")
                continue
            except ValueError as e:
                print(f"Value error processing {finger} in {image_type}: {e}")
                continue
    
    if not hand_data and not sphere_data:
        raise ValueError("No hand or sphere data could be extracted from the XML file")
        
    return {
        'fingers': hand_data,
        'sphere': sphere_data
    }

def draw_hand_skeleton(image, hand_data):
    h, w = image.shape[:2]
    debug_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    colors = {
        'Thumb': (255, 0, 0),    # Blue
        'Index': (0, 255, 0),    # Green
        'Middle': (0, 255, 255), # Yellow
        'Ring': (0, 165, 255),   # Orange
        'Pinky': (0, 0, 255)     # Red
    }
    
    def leap_to_image(point, image_width, image_height, scale=1.5):
        """
        Convert Leap Motion coordinates to image coordinates.
        """
        x_leap, y_leap, z_leap = point
        x_image = int((x_leap * scale) + image_width / 2)
        y_image = int(image_height / 2 - (y_leap * scale))  # Invert Y-axis
        return (x_image, y_image)
    
    """
    # Draw each finger
    for finger_name, finger_data in hand_data['fingers'].items():
        points = [
            leap_to_image(finger_data['tip'], w, h),
            leap_to_image(finger_data['dip'], w, h),
            leap_to_image(finger_data['pip'], w, h),
            leap_to_image(finger_data['mcp'], w, h)
        ]
        
        # Debug: Print the image coordinates
        print(f"{finger_name} Points on Image:")
        for idx, point_label in enumerate(['Tip', 'DIP', 'PIP', 'MCP']):
            print(f"  {point_label}: {points[idx]}")
        
        color = colors.get(finger_name, (255, 255, 255))  # Default to white if not found
        # Draw bones
        for i in range(len(points) - 1):
            cv2.line(debug_img, points[i], points[i + 1], color, 2)
            cv2.circle(debug_img, points[i], 3, color, -1)
        cv2.circle(debug_img, points[-1], 3, color, -1)
    """
    
    # Draw Sphere Center with Radius if available
    if 'sphere' in hand_data and hand_data['sphere']:
        sphere = hand_data['sphere']
        sphere_center = sphere.get('center', None)
        sphere_radius = sphere.get('radius', None)
        
        if sphere_center is not None and sphere_radius is not None:
            # Convert SphereCenter to image coordinates
            sphere_center_img = leap_to_image(sphere_center, w, h)
            # Scale SphereRadius appropriately (You may need to adjust the scale factor)
            sphere_radius_img = int(sphere_radius * 1.5)  # Example scaling
            
            # Draw the sphere (circle) on the debug_img
            cv2.circle(debug_img, sphere_center_img, sphere_radius_img, (255, 255, 255), 2)
            cv2.circle(debug_img, sphere_center_img, 5, (0, 0, 255), -1)  # Draw center point
            
            print(f"Sphere Center on Image: {sphere_center_img}")
            print(f"Sphere Radius on Image: {sphere_radius_img}")
    
    result = cv2.addWeighted(image, 0.7, debug_img, 0.3, 0)
    return result, debug_img

# Main execution
if __name__ == "__main__":
    # Use absolute or correct relative paths
    image_path = "datasets/multimodalkaggle/near-infrared/00/test_gesture/02_l/00/frame_4312_l.png"
    xml_path = "datasets/multimodalkaggle/skeletal/00/test_gesture/02_l/00/frame_4312.xml"

    # Read image and data
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    try:
        hand_data = read_leap_data_et(xml_path)
    except Exception as e:
        print(f"Failed to read hand data: {str(e)}")
        exit(1)

    # Draw skeleton and sphere
    result, debug_img = draw_hand_skeleton(image, hand_data)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Debug Skeleton', debug_img)
    cv2.imshow('Combined Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()