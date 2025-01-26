from PIL import Image
from torch.utils.data import Dataset
import os
import json
from typing import Tuple, Dict, Any
from .eval_predictions import HandLandmarks, create_hand_landmarks_from_model_output, calculate_pck, calculate_iou
import torchvision.transforms as transforms
import numpy as np
import torch

class HandLandmarksDataset(Dataset):
    def __init__(self, image_dir: str, annotations_path: str, transform=None):
        """Initialize the hand landmarks dataset
        
        Args:
            image_dir (str): Directory containing the hand images
            annotations_path (str): Path to JSON file containing landmark annotations
            transform: Optional transform to be applied to images
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        self.annotations = [
            ann for ann in self.annotations 
            if (len(ann.get("landmarks", [])) > 0) 
            or (isinstance(ann.get("rhand_bbox"), list) and len(ann.get("rhand_bbox", [])) == 4 
                and isinstance(ann.get("lhand_bbox"), list) and len(ann.get("lhand_bbox", [])) == 4)
        ]
        # Create mapping of image names to annotations
        self.image_to_annotation: Dict[str, Any] = {
            ann["image"]: ann for ann in self.annotations
        }
        
        # Get list of valid image files
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f in self.image_to_annotation and 
            os.path.isfile(os.path.join(image_dir, f))
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, HandLandmarks]:
        """Get image and its landmarks
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, landmarks) where image is a PIL Image and 
                  landmarks is a HandLandmarks instance
        """
        # Get image name and load image
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Get annotation and create HandLandmarks instance
        annotation = self.image_to_annotation[image_name]
        landmarks = HandLandmarks(annotation, width=image.width, height=image.height)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        return image, landmarks
    def perform_evaluation(self, model, transform=None, model_type="coral"):
        from utils.preprocessing import load_img, process_bbox, generate_patch_image
        from config import cfg
        from tqdm import tqdm
        
        model.eval()
        predictions = []
        annotations = []
        used_image_names = set()
        
        # Add progress bar
        for img_path in tqdm(self.image_files, desc="Evaluating model"):
            image_path = os.path.join(self.image_dir, img_path)
            original_img = load_img(image_path)
            img_height, img_width = original_img.shape[:2]
            annotation = self.image_to_annotation[img_path]
            annotation = HandLandmarks(annotation, width=img_width, height=img_height)
            current_image_name = annotation.image_name
            bbox = [0, 0, img_width, img_height]
            bbox = process_bbox(bbox, img_width, img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            transform = transforms.ToTensor()
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {"source_bb2img_trans": bb2img_trans}
            with torch.no_grad():
                if model_type == "coral":
                    out = model(inputs, {'img': img.clone()}, targets, {}, meta_info, 'test')
                elif model_type == "ssa":
                    out = model(inputs, targets, meta_info, 'test')
            prediction = create_hand_landmarks_from_model_output(out, bb2img_trans, img_width, img_height, current_image_name)
            predictions.append(prediction)
            annotations.append(annotation)
            
        report = None
        for ann in annotations:
            ann.get_lhand_bbox()
            ann.get_rhand_bbox()
            
        if all((ann.landmarks != None and ann.landmarks != []) for ann in annotations):
            report = calculate_pck(predictions, annotations, 2, adaptive_threshold=True, optimal_lr=True)
            
        if all(ann.rhand_bbox != None for ann in annotations) and all(ann.lhand_bbox != None for ann in annotations):
            report = calculate_iou(predictions, annotations, width=img_width, height=img_height, prev_report=report)
            
        return report
