#!/usr/bin/env python3
"""
Perform AdaBN method on a pre-trained InterWild model using a set of images to update BN statistics,
then test on another set of images and save predictions to a JSON file.

"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from glob import glob
from functools import partial

# Adjust these import paths as necessary, depending on your repository structure
# e.g. 
sys.path.insert(0, os.path.join("..", "main"))
sys.path.insert(0, os.path.join("..", "data"))
sys.path.insert(0, os.path.join("..", "common"))

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# InterWild-specific imports (adjust paths if needed):
from config import cfg
from model_new import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.mano import mano
from adabn_utils import BatchNormStatHook, replace_bn_stats  # from your existing adabn_utils

# --------------------------------------------------------------------------------
# Helper dataset class
# --------------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """
    Generic dataset that loads all images from a folder and preprocesses them 
    for the InterWild model.
    """
    def __init__(self, folder_path, input_img_shape=(256, 256)):
        self.image_paths = sorted(glob(os.path.join(folder_path, "*.png"))) + \
                          sorted(glob(os.path.join(folder_path, "*.jpg"))) + \
                          sorted(glob(os.path.join(folder_path, "*.jpeg")))
        self.input_img_shape = input_img_shape
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_img = load_img(img_path)  # InterWild's load_img
        img_height, img_width = original_img.shape[:2]
        
        # Use the entire image as bounding box
        bbox = [0, 0, img_width, img_height]
        bbox = process_bbox(bbox, img_width, img_height)
        
        # Generate cropped, resized image
        img, _, _ = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, self.input_img_shape
        )
        
        img = img.astype(np.float32) / 255.0
        img = self.transform(img)  # shape: [C, H, W]
        return img, os.path.basename(img_path), img_width, img_height


# --------------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------------
def main(args):
    # 1) Set up environment
    cfg.set_args("0")
    cudnn.benchmark = True
    
    # 2) Load the model
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        sys.exit(1)

    print(f"Loading model from {args.model_path} ...")
    model = get_model('test') 
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['network'], strict=False)
    model.eval()

    # 3) Create datasets based on enabled flags
    datasets = []
    print(args)
    if args.update_folder_enabled:
        print(f"Using images from {args.update_folder} for BN statistics update...")
        update_dataset = ImageFolderDataset(args.update_folder, input_img_shape=cfg.input_img_shape)
        datasets.append(update_dataset)
    
    if args.test_folder_enabled:
        print(f"Using images from {args.test_folder} for BN statistics update...")
        test_dataset = ImageFolderDataset(args.test_folder, input_img_shape=cfg.input_img_shape)
        datasets.append(test_dataset)

    if datasets:  # If at least one dataset is enabled
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(datasets)
        combined_loader = DataLoader(
            combined_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0
        )

        # 4) Optionally set BN layers to train mode and set hook
        hook = BatchNormStatHook()

        # Switch BN layers as needed
        if args.train_bn:
            print("Setting BN layers to train mode for AdaBN statistics update.")
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.train()  # put BN layer in train mode
                    module.momentum = args.momentum
                    hook_handle = module.register_forward_hook(partial(hook, name=name))
                    hook.hooks.append(hook_handle)
        else:
            print("BN layers remain in eval mode for AdaBN statistics update.")
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                
                    # Even in eval mode, we can still gather stats via the forward hook if needed
                    module.momentum = args.momentum
                    hook_handle = module.register_forward_hook(partial(hook, name=name))
                    hook.hooks.append(hook_handle)

        # 5) Run inference on combined dataset to gather BN stats
        print(f"Gathering BN statistics using {len(combined_dataset)} images...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for data_batch, _, _, _ in combined_loader:
                data_batch = data_batch.to(device)
                inputs = {'img': data_batch}
                targets = {}
                meta_info = {}
                model(inputs, targets, meta_info, 'test')
        
        # 6) Unhook and finalize BN stats
        hook.unhook()

        # Average BN stats
        for layer_name, stats in hook.bn_stats.items():
            mean = stats['mean'] / stats['count']
            var = stats['var'] / stats['count']
            hook.bn_stats[layer_name] = {'mean': mean, 'var': var}

        if args.post_update:
            print("Updating BN layer parameters with computed AdaBN statistics ...")
            replace_bn_stats(model, hook.bn_stats, ratio=1)
        else:
            print("Skipping BN layer parameter updates in the model.")
    else:
        print("Skipping BN statistics update (no folders enabled)")

    # 8) Test on the test folder (always run final inference)
    print(f"Running final inference in {args.test_folder} and saving results to {args.output_json} ...")

    def normalize_coords(x, y, img_width, img_height):
        return {
            "x": float(x) / img_width,
            "y": float(y) / img_height
        }

    test_folder_img_paths = glob(os.path.join(args.test_folder, "*.jpg")) + \
                           glob(os.path.join(args.test_folder, "*.jpeg")) + \
                           glob(os.path.join(args.test_folder, "*.png"))
    predictions = []
    model.eval()  # Typically test in eval mode
    with torch.no_grad():
        for img_path in test_folder_img_paths:
            # Load and process image (your existing code)
            original_img = load_img(img_path)
            img_height, img_width = original_img.shape[:2]
            bbox = [0, 0, img_width, img_height]
            bbox = process_bbox(bbox, img_width, img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            transform = transforms.ToTensor()
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
                # Forward pass
            inputs = {'img': img}
            targets = {}
            meta_info = {}
            with torch.no_grad():
                out = model(inputs, targets, meta_info, 'test')
                # Get predictions for both hands
            prediction = {
                "image": os.path.basename(img_path),
                "width": int(img_width),
                "height": int(img_height),
                "landmarks": [],
                "normalized": True
            }
            
            for hand in ('left', 'right'):
                # Get joint coordinates
                joint_img = out[hand[0] + 'joint_img'].cpu().numpy()[0]
                joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
                joint_img = np.dot(bb2img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
                
                # Normalize and format coordinates
                hand_points = []
                for point in joint_img:
                    hand_points.append(normalize_coords(point[0], point[1], img_width, img_height))
                
                prediction["landmarks"].append(hand_points)
            
            predictions.append(prediction)
    
    # 9) Save predictions to JSON
    if os.path.dirname(args.output_json) != "":
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(predictions, f, indent=2)

    print("Inference complete. Results saved at:", args.output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform AdaBN on InterWild model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint (e.g. snapshot_0.pth)."
    )
    parser.add_argument(
        "--update_folder",
        type=str,
        required=True,
        help="Path to the folder of images used to update BN statistics."
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        required=True,
        help="Path to the folder of images used for final testing."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to the JSON file where predictions will be saved."
    )
    parser.add_argument(
        "--train_bn",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to put BatchNorm layers in train mode for BN statistic updates."
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.1,
        help="Momentum value to set for all BN layers."
    )
    parser.add_argument(
        "--post_update",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to update BN layers with the newly computed stats after hooking."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size used to run the model on the BN update images."
    )
    parser.add_argument(
        "--update_folder_enabled",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use images from update_folder for BN statistics update."
    )
    parser.add_argument(
        "--test_folder_enabled",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use images from test_folder for BN statistics update."
    )
    args = parser.parse_args()
    main(args)