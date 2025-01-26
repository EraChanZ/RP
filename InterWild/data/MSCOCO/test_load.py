import torch
from torchvision import transforms
import sys
import os
import os.path as osp
import numpy as np

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))

from MSCOCO import MSCOCO

def check_file_exists(filepath):
    if not osp.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    return True

def test_mscoco_dataset():
    # Define basic transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Check required files and directories
    base_path = osp.join('..', 'data', 'MSCOCO')
    required_files = [
        osp.join(base_path, 'annotations', 'coco_wholebody_val_v1.0.json'),
        osp.join(base_path, 'images', 'val2017')
    ]

    # Verify all required files exist
    all_files_exist = True
    for file_path in required_files:
        if not check_file_exists(file_path):
            all_files_exist = False
    
    if not all_files_exist:
        print("\nMissing required files. Please ensure all necessary COCO files are downloaded and in the correct location:")
        print("Expected directory structure:")
        print("InterWild/")
        print("└── data/")
        print("    └── MSCOCO/")
        print("        ├── annotations/")
        print("        │   └── coco_wholebody_val_v1.0.json")
        print("        └── images/")
        print("            └── val2017/")
        return

    try:
        # Initialize dataset in validation mode
        print("\nInitializing MSCOCO dataset...")
        dataset = MSCOCO(transform=transform, data_split='val')
        
        # Print basic information
        print(f"Dataset size: {len(dataset)}")

        # Test loading first item
        print("\nTrying to load first item...")
        inputs, targets, meta_info = dataset[0]
        
        # Print shapes and contents
        print("\nInputs:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={v.shape}")
            else:
                print(f"{k}: type={type(v)}")

        print("\nTargets:")
        for k, v in targets.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}")
            else:
                print(f"{k}: type={type(v)}")

        print("\nMeta Info:")
        for k, v in meta_info.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                print(f"{k}: shape={v.shape}")
            else:
                print(f"{k}: type={type(v)}")

    except Exception as e:
        print(f"\nError occurred while loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mscoco_dataset()