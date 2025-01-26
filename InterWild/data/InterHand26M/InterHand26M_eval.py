import os
import os.path as osp
import numpy as np
import torch
import cv2
from utils.preprocessing import load_img, process_bbox, augmentation

class InterHand26M_eval(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, offset=0, size=None):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'InterHand26M', 'images')
        self.offset = offset
        self.size = size
        
        # Get list of all images in the directory
        self.datalist = self.load_data()

    def total_files(self):
        total_files = 0
        for root, _, files in os.walk(osp.join(self.img_path, self.data_split)):
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                total_files += 1
        return total_files

    def load_data(self):
        datalist = []
        current_count = 0
        skip_count = 0
        
        # Walk through the images directory
        for root, _, files in os.walk(osp.join(self.img_path, self.data_split)):
            for file in files:
                if not file.endswith(('.jpg', '.png')):
                    continue
                    
                # Skip files before offset
                if skip_count < self.offset:
                    skip_count += 1
                    continue
                    
                # Stop if we've reached size limit
                if self.size is not None and current_count >= self.size:
                    return datalist
                
                img_path = osp.join(root, file)
                
                # Load image to get dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img_height, img_width = img.shape[:2]
                body_bbox = np.array([0, 0, img_width, img_height], dtype=np.float32)
                body_bbox = process_bbox(body_bbox, img_width, img_height, extend_ratio=1.0)
                
                if body_bbox is None:
                    continue
                
                datalist.append({
                    'img_path': img_path,
                    'img_shape': (img_height, img_width),
                    'body_bbox': body_bbox
                })
                
                current_count += 1
        
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, body_bbox = data['img_path'], data['body_bbox']

        # Load and process image
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        # Create empty targets and meta_info
        inputs = {'img': img}
        targets = {}
        meta_info = {'bb2img_trans': bb2img_trans}

        return inputs, targets, meta_info