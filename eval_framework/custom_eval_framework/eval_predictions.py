from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class HandLandmarks:
    def __init__(self, json_data, width=None, height=None):
        """Initialize hand landmarks from JSON data
        
        Args:
            json_data (dict): Dictionary containing image name and landmarks data
            width (int, optional): Image width for denormalization. Defaults to None.
            height (int, optional): Image height for denormalization. Defaults to None.
        """
        self.image_name = json_data.get("image", "")
        self.width = json_data.get("width", None) if width is None else width
        self.height = json_data.get("height", None) if height is None else height
        self.normalized = json_data.get("normalized", True)  # Default to normalized if not specified
        self.landmarks = []
        
        # Add bounding box storage
        self.rhand_bbox = json_data.get("rhand_bbox", None)
        self.lhand_bbox = json_data.get("lhand_bbox", None)
        
        # Process landmarks arrays
        for landmark_set in json_data.get("landmarks", []):
            points = []
            for point in landmark_set:
                x = point.get("x", 0.0)
                y = point.get("y", 0.0)
                points.append((x, y))
            self.landmarks.append(points)

    @property
    def left_hand(self):
        """Get left hand landmarks if available"""
        return self.landmarks[0] if len(self.landmarks) > 0 else []

    @property 
    def right_hand(self):
        """Get right hand landmarks if available"""
        return self.landmarks[1] if len(self.landmarks) > 1 else []

    def __len__(self):
        """Return number of hand landmark sets"""
        return len(self.landmarks)

    def normalize(self):
        """Convert coordinates to normalized (0-1) range"""
        if self.normalized or not (self.width and self.height):
            return

        # Normalize landmarks
        for hand_idx, hand in enumerate(self.landmarks):
            normalized_points = []
            for x, y in hand:
                norm_x = x / self.width
                norm_y = y / self.height
                normalized_points.append((norm_x, norm_y))
            self.landmarks[hand_idx] = normalized_points
        
        # Normalize bounding boxes if they exist
        if self.rhand_bbox is not None:
            x1, y1, x2, y2 = self.rhand_bbox
            self.rhand_bbox = [x1/self.width, y1/self.height, x2/self.width, y2/self.height]
            
        if self.lhand_bbox is not None:
            x1, y1, x2, y2 = self.lhand_bbox
            self.lhand_bbox = [x1/self.width, y1/self.height, x2/self.width, y2/self.height]
        
        self.normalized = True

    def denormalize(self):
        """Convert normalized coordinates to pixel coordinates"""
        if not self.normalized or not (self.width and self.height):
            return

        # Denormalize landmarks
        for hand_idx, hand in enumerate(self.landmarks):
            pixel_points = []
            for x, y in hand:
                px = x * self.width
                py = y * self.height
                pixel_points.append((px, py))
            self.landmarks[hand_idx] = pixel_points
        
        # Denormalize bounding boxes if they exist
        if self.rhand_bbox is not None:
            x1, y1, x2, y2 = self.rhand_bbox
            self.rhand_bbox = [x1*self.width, y1*self.height, x2*self.width, y2*self.height]
            
        if self.lhand_bbox is not None:
            x1, y1, x2, y2 = self.lhand_bbox
            self.lhand_bbox = [x1*self.width, y1*self.height, x2*self.width, y2*self.height]
        
        self.normalized = False

    def set_image_size(self, width, height):
        """Set image dimensions for coordinate conversion
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels
        """
        self.width = width
        self.height = height
    def visualize(self, image_dir):
       """Visualize hand landmarks on the original image
       
       Args:
           image_dir (str): Directory containing the original image
           
       Returns:
           PIL.Image: Image with visualized landmarks
           
       Raises:
           FileNotFoundError: If image file is not found
           ValueError: If image_name is not set
       """
       
       if not self.image_name:
           raise ValueError("Image name is not set")
           
       image_path = os.path.join(image_dir, self.image_name)
       if not os.path.exists(image_path):
           raise FileNotFoundError(f"Image not found: {image_path}")
           
       # Load and convert image to RGB
       image = Image.open(image_path).convert('RGB')
       draw = ImageDraw.Draw(image)
       
       # Ensure coordinates are in pixel space
       original_normalized = self.normalized
       if self.normalized:
           self.set_image_size(image.width, image.height)
           self.denormalize()
           
       # Define visualization parameters
       point_radius = min(image.width, image.height) // 300
       font_size = point_radius * 2
       try:
           font = ImageFont.truetype("arial.ttf", font_size)
       except OSError:
           font = ImageFont.load_default()
           
       # Draw landmarks for each hand
       for hand_idx, hand in enumerate(self.landmarks):
           # Blue for left hand (index 0), Red for right hand (index 1)
           color = (0, 0, 255) if hand_idx == 0 else (255, 0, 0)
           
           for point_idx, (x, y) in enumerate(hand):
               # Draw point
               x, y = int(x), int(y)
               draw.ellipse(
                   [(x - point_radius, y - point_radius),
                    (x + point_radius, y + point_radius)],
                   fill=color
               )
               
               # Draw point index
               text = str(point_idx)
               draw.text(
                   (x + point_radius, y - point_radius),
                   text,
                   fill=color,
                   font=font
               )
       
       # Restore original normalization state
       if original_normalized:
           self.normalize()
           
       return image

    def get_rhand_bbox(self):
        """Get right hand bounding box
        
        Returns:
            list[float]: [x1, y1, x2, y2] coordinates or None if no right hand data
        """
        # Return stored bbox if it exists
        if self.rhand_bbox is not None:
            return self.rhand_bbox
            
        # Calculate from landmarks if right hand exists
        if len(self.landmarks) > 1 and self.landmarks[1]:  # Right hand is index 1
            x_coords = [p[0] for p in self.landmarks[1]]
            y_coords = [p[1] for p in self.landmarks[1]]
            self.rhand_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            return self.rhand_bbox
            
        return None
        
    def get_lhand_bbox(self):
        """Get left hand bounding box
        
        Returns:
            list[float]: [x1, y1, x2, y2] coordinates or None if no left hand data
        """
        # Return stored bbox if it exists
        if self.lhand_bbox is not None:
            return self.lhand_bbox
            
        # Calculate from landmarks if left hand exists
        if len(self.landmarks) > 0 and self.landmarks[0]:  # Left hand is index 0
            x_coords = [p[0] for p in self.landmarks[0]]
            y_coords = [p[1] for p in self.landmarks[0]]
            self.lhand_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]  
            return self.lhand_bbox
            
        return None

def align_predictions_with_labels(predictions, labels):
   """Align predictions with ground truth labels by image name
   
   Args:
       predictions (list[HandLandmarks]): List of predicted hand landmarks
       labels (list[HandLandmarks]): List of ground truth hand landmarks
       
   Returns:
       tuple[list[HandLandmarks], list[HandLandmarks]]: Aligned predictions and labels
       
   Raises:
       ValueError: If lists cannot be properly aligned or contain duplicates
   """
   # Check for duplicates in both lists
   pred_names = [p.image_name for p in predictions]
   label_names = [l.image_name for l in labels]
   
   pred_duplicates = {name for name in pred_names if pred_names.count(name) > 1}
   label_duplicates = {name for name in label_names if label_names.count(name) > 1}
   
   if pred_duplicates:
       raise ValueError(f"Duplicate image names found in predictions: {pred_duplicates}")
   if label_duplicates:
       raise ValueError(f"Duplicate image names found in labels: {label_duplicates}")
   
   # Convert to dictionaries for easier lookup
   pred_dict = {p.image_name: p for p in predictions}
   label_dict = {l.image_name: l for l in labels}
   
   # Find common image names
   common_names = set(pred_dict.keys()) & set(label_dict.keys())
   
   # Check if any items will be unpaired
   missing_preds = set(label_dict.keys()) - common_names
   missing_labels = set(pred_dict.keys()) - common_names
   
   if missing_preds or missing_labels:
       raise ValueError(
           f"Unmatched items found:\n"
           f"Images missing predictions: {missing_preds}\n"
           f"Images missing labels: {missing_labels}"
       )
   
   # Sort by image name for consistent ordering
   aligned_names = sorted(common_names)
   aligned_preds = [pred_dict[name] for name in aligned_names]
   aligned_labels = [label_dict[name] for name in aligned_names]
   
   return aligned_preds, aligned_labels


def load_predictions(json_path, width=None, height=None, image_dir=None):
   """Load hand landmarks predictions from JSON file
   
   Args:
       json_path (str): Path to JSON file containing predictions
       width (int, optional): Image width for denormalization. Defaults to None.
       height (int, optional): Image height for denormalization. Defaults to None.
       image_dir (str, optional): Directory containing images. Used to get dimensions if width/height not provided.
       
   Returns:
       list[HandLandmarks]: List of HandLandmarks objects
   """
   import json
   from PIL import Image
   import os
   
   with open(json_path, 'r') as f:
       predictions = json.load(f)
   
   landmarks_list = []
   for pred in predictions:
       img_width, img_height = width, height
       if (width is None or height is None) and image_dir is not None:
           try:
               image_path = os.path.join(image_dir, pred["image"])
               with Image.open(image_path) as img:
                   img_width, img_height = img.size
           except:
               pass
               
       hand = HandLandmarks(pred, width=img_width, height=img_height)
       landmarks_list.append(hand)
       
   return landmarks_list

def skip_predictions_without_labels(predictions, labels):
    """Skip predictions without corresponding labels
    
    Args:
        predictions (list[HandLandmarks]): List of predicted hand landmarks
        labels (list[HandLandmarks]): List of ground truth hand landmarks
        
    Returns:
        tuple[list[HandLandmarks], list[HandLandmarks]]: Aligned predictions and labels
    """
    return [p for p in predictions if p.image_name in [l.image_name for l in labels]]

def align_norm_predictions_with_labels(predictions, labels):
    """Align predictions with ground truth labels by image name and ensure proper denormalization
    
    Args:
        predictions (list[HandLandmarks]): List of predicted hand landmarks
        labels (list[HandLandmarks]): List of ground truth hand landmarks
        
    Returns:
        tuple[list[HandLandmarks], list[HandLandmarks]]: Aligned and denormalized predictions and labels
        
    Raises:
        ValueError: If lists cannot be properly aligned, contain duplicates, or lack image dimensions
    """
    # First perform basic alignment
    aligned_preds, aligned_labels = align_predictions_with_labels(predictions, labels)
    
    # Check and handle normalization for each pair
    for pred, label in zip(aligned_preds, aligned_labels):
        # Check if either has dimensions set
        pred_has_dims = pred.width is not None and pred.height is not None
        label_has_dims = label.width is not None and label.height is not None
        
        if not (pred_has_dims or label_has_dims):
            raise ValueError(
                f"Neither prediction nor label for image '{pred.image_name}' "
                "has width and height dimensions set"
            )
        
        # If one has dimensions and other doesn't, copy dimensions
        if pred_has_dims and not label_has_dims:
            label.set_image_size(pred.width, pred.height)
        elif label_has_dims and not pred_has_dims:
            pred.set_image_size(label.width, label.height)
            
        # Ensure both are denormalized
        pred.denormalize()
        label.denormalize()
    
    return aligned_preds, aligned_labels

def calculate_pck(predictions, labels, threshold, width=None, height=None, adaptive_threshold=False, optimal_lr=False, prev_report=None, skip_without_labels=False, image_dir=None):
    """Calculate Percentage of Correct Keypoints (PCK) metric
   
    Args:
        predictions: Either path to JSON file or list of HandLandmarks
        labels: Either path to JSON file or list of HandLandmarks
        threshold (float): Distance threshold as ratio of min(width, height), or multiplier for adaptive threshold
        width (int, optional): Image width for denormalization. Defaults to None.
        height (int, optional): Image height for denormalization. Defaults to None.
        adaptive_threshold (bool, optional): If True, calculate threshold from label keypoint distances. Defaults to False.
        optimal_lr (bool, optional): If True, try both L/R hand combinations and select best score. Defaults to False.
       
    Returns:
        EvaluationReport: Report containing PCK scores and visualizations
    """
    # Load and align data as before
    if isinstance(predictions, str):
        predictions = load_predictions(predictions, width, height, image_dir)
    if isinstance(labels, str):
        labels = load_predictions(labels, width, height, image_dir)
       
    if skip_without_labels:
        predictions = skip_predictions_without_labels(predictions, labels)
       
    pred_aligned, label_aligned = align_norm_predictions_with_labels(predictions, labels)
    
    if prev_report is not None:
        report = prev_report
    else:
        report = EvaluationReport()

    # Calculate PCK for each image
    for pred, label in zip(pred_aligned, label_aligned):
        image_correct = 0
        image_total = 0
        
        # Calculate adaptive threshold for this specific label
        if adaptive_threshold:
            # Find minimum distance between any two points in the label
            min_dist = float('inf')
            for hand in label.landmarks:
                if len(hand) < 2:  # Skip hands with less than 2 points
                    continue
                    
                # Calculate distances between all pairs of points
                for i, (x1, y1) in enumerate(hand):
                    for j, (x2, y2) in enumerate(hand):
                        if i != j:
                            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                            min_dist = min(min_dist, dist)
            
            if min_dist == float('inf'):
                raise ValueError(f"No valid keypoint pairs found for adaptive threshold calculation in image {pred.image_name}")
                
            distance_threshold = threshold * (min_dist / 2)
        else:
            min_dim = min(pred.width, pred.height)
            distance_threshold = threshold * min_dim

        if optimal_lr and len(pred.landmarks) == 2 and len(label.landmarks) == 2:
            # Try both hand combinations and select the best one
            scores = []
            
            # Original order
            correct1, total1 = 0, 0
            for hand_idx in range(2):
                pred_hand = pred.landmarks[hand_idx]
                label_hand = label.landmarks[hand_idx]
                for pred_point, label_point in zip(pred_hand, label_hand):
                    pred_x, pred_y = pred_point
                    label_x, label_y = label_point
                    distance = ((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2) ** 0.5
                    if distance <= distance_threshold:
                        correct1 += 1
                    total1 += 1
            
            # Swapped order
            correct2, total2 = 0, 0
            for hand_idx in range(2):
                pred_hand = pred.landmarks[1 - hand_idx]  # Swap hands
                label_hand = label.landmarks[hand_idx]
                for pred_point, label_point in zip(pred_hand, label_hand):
                    pred_x, pred_y = pred_point
                    label_x, label_y = label_point
                    distance = ((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2) ** 0.5
                    if distance <= distance_threshold:
                        correct2 += 1
                    total2 += 1
            
            # Use the better score
            if correct1/total1 >= correct2/total2:
                image_correct, image_total = correct1, total1
            else:
                image_correct, image_total = correct2, total2
                # Swap the hands in the prediction for visualization
                pred.landmarks[0], pred.landmarks[1] = pred.landmarks[1], pred.landmarks[0]
        else:
            # Original PCK calculation logic
            for hand_idx in range(min(len(pred), len(label))):
                pred_hand = pred.landmarks[hand_idx]
                label_hand = label.landmarks[hand_idx]
                
                for pred_point, label_point in zip(pred_hand, label_hand):
                    pred_x, pred_y = pred_point
                    label_x, label_y = label_point
                    distance = ((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2) ** 0.5
                    if distance <= distance_threshold:
                        image_correct += 1
                    image_total += 1
        
        if image_total > 0:
            image_score = image_correct / image_total
            report.add_pck_result(pred, label, image_score, distance_threshold)
   
    return report

def calculate_bbox_iou(bboxA, bboxB):
    """
    Compute the Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        bboxA (list or tuple): [x1, y1, x2, y2]
        bboxB (list or tuple): [x1, y1, x2, y2]
        
    Returns:
        float: IoU between 0.0 and 1.0
    """
    # If either bounding box is None or empty, return 0
    if bboxA is None or bboxB is None:
        return 0.0
    
    # Unpack coordinates
    xA1, yA1, xA2, yA2 = bboxA
    xB1, yB1, xB2, yB2 = bboxB
    
    # Ensure boxes are properly ordered (x1 < x2, y1 < y2). If not, correct them
    xA1, xA2 = sorted([xA1, xA2])
    yA1, yA2 = sorted([yA1, yA2])
    xB1, xB2 = sorted([xB1, xB2])
    yB1, yB2 = sorted([yB1, yB2])
    
    # Calculate intersection
    interX1 = max(xA1, xB1)
    interY1 = max(yA1, yB1)
    interX2 = min(xA2, xB2)
    interY2 = min(yA2, yB2)
    
    interW = max(0.0, interX2 - interX1)
    interH = max(0.0, interY2 - interY1)
    interArea = interW * interH
    
    # Calculate each box's area
    boxAArea = max(0.0, xA2 - xA1) * max(0.0, yA2 - yA1)
    boxBArea = max(0.0, xB2 - xB1) * max(0.0, yB2 - yB1)
    
    # Compute IoU
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    
    return interArea / unionArea


def calculate_iou(predictions, labels, width=None, height=None, prev_report=None, skip_without_labels=False, image_dir=None):
    """
    Calculate Intersection over Union (IoU) metric for right and left hand bounding boxes.
    
    This function loads the data (if given as paths), aligns predictions and labels, and
    calculates the IoU for each image's right and left hand bounding boxes. An average IoU
    across both hands (if both exist) is recorded per image.
    
    Args:
        predictions: Either path to JSON file or list of HandLandmarks
        labels: Either path to JSON file or list of HandLandmarks
        width (int, optional): Image width for denormalization. Defaults to None.
        height (int, optional): Image height for denormalization. Defaults to None.
        skip_without_labels (bool, optional): If True, skip predictions without corresponding labels. Defaults to False.

    Returns:
        EvaluationReport: Report containing IoU scores and visualizations
    """
    # Load and align data similarly to calculate_pck
    if isinstance(predictions, str):
        predictions = load_predictions(predictions, width, height, image_dir)
    if isinstance(labels, str):
        labels = load_predictions(labels, width, height, image_dir)
    
    if skip_without_labels:
        predictions = skip_predictions_without_labels(predictions, labels)  
       
    pred_aligned, label_aligned = align_norm_predictions_with_labels(predictions, labels)
    
    if prev_report is not None:
        report = prev_report
    else:
        report = EvaluationReport()
    
    for pred, label in zip(pred_aligned, label_aligned):
        # Get right-hand and left-hand bboxes
        if pred.normalized:
            pred.denormalize()
        if label.normalized:
            label.denormalize()

        pred_rbbox = pred.get_rhand_bbox()
        label_rbbox = label.get_rhand_bbox()
        pred_lbbox = pred.get_lhand_bbox()
        label_lbbox = label.get_lhand_bbox()
        print(pred_rbbox, label_rbbox)
        # Calculate IoUs
        r_iou = calculate_bbox_iou(pred_rbbox, label_rbbox)
        l_iou = calculate_bbox_iou(pred_lbbox, label_lbbox)
        
        # If both hands exist, average them. Otherwise, just use the one that exists
        ious = []
        if pred_rbbox is not None and label_rbbox is not None:
            ious.append(r_iou)
        if pred_lbbox is not None and label_lbbox is not None:
            ious.append(l_iou)
        
        if len(ious) > 0:
            image_score = sum(ious) / len(ious)
        else:
            # No valid bounding boxes found – mark IoU as 0 for this image
            image_score = 0.0
        
        report.add_iou_result(pred, label, image_score)
    
    return report

@dataclass
class MetricResult:
   """Store metric result for a single prediction-label pair"""
   metric_name: str
   value: float
   parameters: Dict[str, Any]
   prediction: HandLandmarks
   label: HandLandmarks

class EvaluationReport:
    def __init__(self):
        """Initialize empty evaluation report"""
        self.results: List[MetricResult] = []
        
    def __str__(self) -> str:
        """Return string representation with basic statistics"""
        if not self.results:
            return "EvaluationReport: No results"
            
        # Group results by metric name
        metrics = {}
        for result in self.results:
            if result.metric_name not in metrics:
                metrics[result.metric_name] = []
            metrics[result.metric_name].append(result.value)
            
        # Build stats strings
        stats = []
        for metric_name, values in metrics.items():
            mean = sum(values) / len(values)
            stats.append(f"{metric_name}:")
            stats.append(f"  Mean: {mean:.3f}")
            stats.append(f"  Min:  {min(values):.3f}")
            stats.append(f"  Max:  {max(values):.3f}")
            stats.append(f"  Count: {len(values)}")
            
        return "EvaluationReport:\n" + "\n".join(stats)
        
    def add_pck_result(self, prediction: HandLandmarks, label: HandLandmarks, 
                        score: float, threshold: float):
        """Add PCK evaluation result
        
        Args:
            prediction: Prediction HandLandmarks
            label: Label HandLandmarks
            score: PCK score for this pair
            threshold: PCK threshold used
        """
        self.results.append(MetricResult(
            metric_name="PCK",
            value=score,
            parameters={"threshold": threshold},
            prediction=prediction,
            label=label
        ))

    def add_iou_result(self, prediction: HandLandmarks, label: HandLandmarks, score: float):
        """
        Add IoU evaluation result
        """
        self.results.append(MetricResult(
            metric_name="IOU",
            value=score,
            parameters={},
            prediction=prediction,
            label=label
        ))

    def get_sorted_indices(self, metric_name: str, ascending: bool = True) -> List[int]:
       """Get indices of results sorted by metric value"""
       metric_results = [r for r in self.results if r.metric_name == metric_name]
       if not metric_results:
           raise ValueError(f"No results found for metric: {metric_name}")
       
       return sorted(range(len(metric_results)), 
                    key=lambda i: metric_results[i].value,
                    reverse=not ascending)
    def nth_best(self, n: int, image_dir: str = None, metric_name: str = "PCK") -> Image:
       """Visualize the nth best result for a given metric"""
       sorted_idx = self.get_sorted_indices(metric_name, ascending=False)
       if n > len(sorted_idx):
           raise ValueError(f"n ({n}) is larger than number of results ({len(sorted_idx)})")
       return self._visualize_comparison(self.results[sorted_idx[n-1]], image_dir)
    def nth_worst(self, n: int, image_dir: str = None, metric_name: str = "PCK") -> Image:
       """Visualize the nth worst result for a given metric"""
       sorted_idx = self.get_sorted_indices(metric_name, ascending=True)
       if n > len(sorted_idx):
           raise ValueError(f"n ({n}) is larger than number of results ({len(sorted_idx)})")
       return self._visualize_comparison(self.results[sorted_idx[n-1]], image_dir)
    def _visualize_comparison(self, result: MetricResult, image_dir: str = None) -> Image:
       """Create visualization comparing prediction and label"""
       # This method remains the same. It can be used to visualize bounding boxes or points as needed.
       # For IoU, users may want to extend or adapt the drawing to show bounding boxes as well.
       
       # Ensure we have image dimensions
       if not result.prediction.width or not result.prediction.height:
           raise ValueError("Prediction lacks image dimensions")
           
       # Create or load base image
       if image_dir and result.prediction.image_name:
           try:
               img_path = os.path.join(image_dir, result.prediction.image_name)
               img = Image.open(img_path).convert('RGB')
           except (FileNotFoundError, OSError):
               print(f"Warning: Could not load image {img_path}, using blank background")
               img = Image.new('RGB', (result.prediction.width, result.prediction.height), 'white')
       else:
           img = Image.new('RGB', (result.prediction.width, result.prediction.height), 'white')
       
       draw = ImageDraw.Draw(img)
       
       # If visualizing PCK or keypoints
       if result.metric_name == "PCK":
           for hand_idx in range(min(len(result.prediction), len(result.label))):
               pred_hand = result.prediction.landmarks[hand_idx]
               label_hand = result.label.landmarks[hand_idx]
               
               pred_color = (0, 0, 255)
               label_color = (0, 255, 0)
               line_color = (128, 128, 128)
               
               # Optionally draw threshold circle:
               threshold = result.parameters.get("threshold", 0)
               radius = int(threshold)
               for lx, ly in label_hand:
                   draw.ellipse(
                       [(lx - radius, ly - radius), (lx + radius, ly + radius)],
                       outline=(200, 200, 200)
                   )
               
               for (px, py), (lx, ly) in zip(pred_hand, label_hand):
                   draw.line([(px, py), (lx, ly)], fill=line_color, width=3)
                   point_radius = 9
                   draw.ellipse(
                       [(px - point_radius, py - point_radius),
                        (px + point_radius, py + point_radius)],
                       fill=pred_color
                   )
                   draw.ellipse(
                       [(lx - point_radius, ly - point_radius),
                        (lx + point_radius, ly + point_radius)],
                       fill=label_color
                   )

       # If visualizing IoU bounding boxes
       elif result.metric_name == "IOU":
           # We can optionally draw the bounding boxes for R & L if users desire
           # For demonstration, we'll show them in different colors
           pred_r = result.prediction.get_rhand_bbox()
           label_r = result.label.get_rhand_bbox()
           pred_l = result.prediction.get_lhand_bbox()
           label_l = result.label.get_lhand_bbox()
           
           def draw_bbox(bbox, color):
               if bbox is None:
                   return
               x1, y1, x2, y2 = bbox
               draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
           
           # Red for prediction, Green for label
           draw_bbox(pred_r, (255, 0, 0))
           draw_bbox(label_r, (0, 255, 0))
           draw_bbox(pred_l, (255, 0, 0))
           draw_bbox(label_l, (0, 255, 0))
       
       return img

def create_hand_landmarks_from_model_output(model_output: Dict, bb2img_trans, img_width: int, img_height: int, image_name: str) -> HandLandmarks:
    """Convert model output to HandLandmarks instance
    
    Args:
        model_output (Dict): Model output containing joints and optionally bounding boxes
        bb2img_trans: Transformation matrix from bounding box to image coordinates
        img_width (int): Original image width
        img_height (int): Original image height
        image_name (str): Name of the image file
        
    Returns:
        HandLandmarks: Instance containing processed hand landmarks
    """
    # Prepare the base structure for HandLandmarks
    landmarks_dict = {
        "width": int(img_width),
        "height": int(img_height),
        "landmarks": [],
        "normalized": True,
        "image": image_name
    }
    
    scale_bbox_x = 1 / 192
    scale_bbox_y = 1 / 256

    # Process bounding boxes if they exist
    for bbox_key in ['rhand_bbox', 'lhand_bbox']:
        if bbox_key in model_output:
            bbox = model_output[bbox_key].cpu().numpy()[0]  # Assuming shape [1, 4]
            x1, y1, x2, y2 = bbox
            # Scale bbox to image dimensions
            scaled_bbox = [
                float(x1 * scale_bbox_x),
                float(y1 * scale_bbox_y),
                float(x2 * scale_bbox_x),
                float(y2 * scale_bbox_y)
            ]
            landmarks_dict[bbox_key] = scaled_bbox
    
    # Process right and left hand joints
    for hand_prefix in ['r', 'l']:
        # Find the matching joint key in model output
        joint_key = next((k for k in model_output.keys() if f"{hand_prefix}joint_img" in k), None)
        if joint_key is None:
            continue
            
        # Get joint coordinates and convert to numpy
        joint_img = model_output[joint_key].cpu().numpy()[0]
        
        # Add homogeneous coordinate
        joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])), 1)
        
        # Transform to image coordinates
        joint_img = np.dot(bb2img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        
        # Normalize coordinates
        hand_points = []
        for point in joint_img:
            x, y = point[0] / img_width, point[1] / img_height
            hand_points.append({"x": float(x), "y": float(y)})
            
        landmarks_dict["landmarks"].append(hand_points)
    
    # Create and return HandLandmarks instance
    return HandLandmarks(landmarks_dict)

def calculate_coral_train_loss(source_dataset, target_dataset, batch_size: int, model, prev_report=None) -> EvaluationReport:
    """Calculate CORAL training loss between source and target datasets
    
    Args:
        source_dataset (CustomHandLandmarksDataset): Source domain dataset
        target_dataset (CustomHandLandmarksDataset): Target domain dataset
        batch_size (int): Batch size for processing
        model: CORAL model in training mode
        
    Returns:
        EvaluationReport: Report containing CORAL and other training losses
    """
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm
    
    # Create data loaders with the custom dataset format
    source_loader = DataLoader(
        dataset=source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    # Initialize report
    report = prev_report if prev_report is not None else EvaluationReport()
    
    # Ensure model is in train mode
    model.train()
    
    # Create iterator for target loader that can be cycled
    target_iter = iter(target_loader)
    
    # Process all batches
    with torch.no_grad():  # No need for gradients since we're just calculating losses
        for source_batch in tqdm(source_loader, desc="Calculating CORAL losses"):
            try:
                target_batch = next(target_iter)
            except StopIteration:
                # Restart target iterator if exhausted
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            # Unpack batches - now using the CustomHandLandmarksDataset format
            source_inputs, source_targets, source_meta_info = source_batch
            target_inputs, target_targets, target_meta_info = target_batch
            
            # Move to GPU if available
            device = next(model.parameters()).device
            
            # Move inputs to device
            for k in source_inputs:
                source_inputs[k] = source_inputs[k].to(device)
            for k in target_inputs:
                target_inputs[k] = target_inputs[k].to(device)
                
            # Move targets to device
            for k in source_targets:
                source_targets[k] = source_targets[k].to(device)
            for k in target_targets:
                target_targets[k] = target_targets[k].to(device)
                
            # Move meta info to device
            # Combine source and target meta info into one dictionary with prefixes
            meta_info = {}
            for k in source_meta_info:
                if isinstance(source_meta_info[k], torch.Tensor):
                    meta_info[f'source_{k}'] = source_meta_info[k].to(device)
            for k in target_meta_info:
                if isinstance(target_meta_info[k], torch.Tensor):
                    meta_info[f'target_{k}'] = target_meta_info[k].to(device)
            # Forward pass in train mode
            losses = model(source_inputs, target_inputs, source_targets, target_targets, meta_info, 'train', only_coral_loss=True)
            
            # Record batch losses
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    if loss_value.numel() > 1:
                        loss_value = loss_value.mean()
                    loss_value = float(loss_value.cpu().numpy())
                report.results.append(MetricResult(
                    metric_name=loss_name,
                    value=loss_value,
                    parameters={},
                    prediction=None,  # Not relevant for loss calculation
                    label=None
                ))

    
    return report

def merge_evaluation_reports(report1: EvaluationReport, report2: EvaluationReport) -> EvaluationReport:
    """Merge two EvaluationReports into a single combined report
    
    Args:
        report1: First evaluation report to merge
        report2: Second evaluation report to merge
        
    Returns:
        EvaluationReport: New report containing combined results from both inputs
    """
    merged_report = EvaluationReport()
    
    # Simple concatenation works since all metric results are independent
    merged_report.results = report1.results + report2.results
    
    return merged_report
