import torch
from torch.utils.data import DataLoader
import os
import os.path as osp
import sys
from torch.utils.data import DataLoader, Dataset
import gc
import numpy as np
from tqdm import tqdm
from torchvision import transforms

class MSCOCODataset(Dataset):
    """
    Adapter for MSCOCO dataset that provides the same interface as InterHand2D.
    Only uses validation split.
    """
    def __init__(self, transform=None):
        from MSCOCO_orig import MSCOCO
        self.dataset = MSCOCO(transform=transform, data_split='train')
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, targets, meta_info = self.dataset[idx]
        
        # Ensure we return the expected format:
        # source_inputs = {'img': tensor}
        # source_targets = {'joint_img', 'bbox' related fields}
        # source_meta_info = validation related fields
        return inputs, targets, meta_info


if __name__ == '__main__':

    
    sys.path.insert(0, osp.join('..', 'main'))
    sys.path.insert(0, osp.join('..', 'data'))
    sys.path.insert(0, osp.join('..', 'common'))

    from config import cfg
    from model_new_ssa import get_model
    from utils.preprocessing import load_img, process_bbox, generate_patch_image, get_iou
    from utils.vis import vis_keypoints_with_skeleton, save_obj, render_mesh_orthogonal
    from utils.mano import mano
    from InterHand26M_eval import InterHand26M_eval
    # Create output directories
    OUTPUT_DIR = r"D:\datasets\saved_features"
    IH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "interhand")
    COCO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mscoco")
    os.makedirs(IH_OUTPUT_DIR, exist_ok=True)
    os.makedirs(COCO_OUTPUT_DIR, exist_ok=True)

    model = get_model(mode='test')
    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(r"C:\Users\vladi\RP\InterWild\demo\snapshot_6.pth")
    load_info = model.load_state_dict(ckpt['network'], strict=False)
    model.eval()

    BATCH_SIZE = 16
    CHUNK_SIZE = 16384 // 2
    SAMPLE_ACCUMULATION = 1024 * 8

    def process_dataset(dataloader, model, running_means=None, dataset_name=""):
        print(f"\nProcessing {dataset_name} dataset...")
        
        if running_means is None:
            running_means = {
                'hand_feat': None,
                'body_feat': None,
                'total_samples': 0
            }
        
        for batch_idx, (inputs, targets, meta_info) in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
            with torch.no_grad():
                # Move input to GPU
                batch_imgs = inputs['img'].cuda()
                outputs = model({'img': batch_imgs}, {}, meta_info, 'test')
                
                # Move features to CPU and convert to numpy
                hand_feats = outputs['source_hand_feat'].cpu().numpy()
                body_feats = outputs['source_body_feat'].cpu().numpy()
                
                # Update running means
                batch_size = hand_feats.shape[0]
                if running_means['hand_feat'] is None:
                    running_means['hand_feat'] = np.zeros_like(hand_feats[0])
                    running_means['body_feat'] = np.zeros_like(body_feats[0])
                
                # Update means using the running average formula
                weight = batch_size / (running_means['total_samples'] + batch_size)
                running_means['hand_feat'] = (1 - weight) * running_means['hand_feat'] + weight * np.mean(hand_feats, axis=0)
                running_means['body_feat'] = (1 - weight) * running_means['body_feat'] + weight * np.mean(body_feats, axis=0)
                running_means['total_samples'] += batch_size
            
            # Clear some memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Add debug print every 100 batches
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx * BATCH_SIZE}/{len(dataloader.dataset)} samples")
                print(f"Memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        return running_means

    # Initialize single global running mean
    global_running_means = None

    # Process InterHand dataset in chunks
    total_ih_samples = 1361062
    num_chunks = (total_ih_samples // CHUNK_SIZE) + 1

    for offset in tqdm(range(0, total_ih_samples, CHUNK_SIZE), desc="Processing InterHand chunks", total=num_chunks):
        chunk_num = offset//CHUNK_SIZE + 1
        print(f"\nProcessing InterHand chunk {chunk_num}/{num_chunks}")
        print(f"Offset: {offset}, Size: {min(CHUNK_SIZE, total_ih_samples - offset)}")
        
        # Load chunk of dataset
        ih_ds = InterHand26M_eval(
            transform=transforms.ToTensor(),
            data_split='train',
            offset=offset,
            size=min(CHUNK_SIZE, total_ih_samples - offset)
        )
        
        # Create dataloader
        ih_loader = DataLoader(
            ih_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Process chunk and update global running means
        global_running_means = process_dataset(ih_loader, model, global_running_means, dataset_name=f"InterHand chunk {chunk_num}")
        
        # Add memory usage print
        print(f"GPU Memory after chunk: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Clear memory
        del ih_ds, ih_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Process MSCOCO dataset
    print("\nProcessing MSCOCO dataset")
    mscoco_ds = MSCOCODataset(transform=transforms.ToTensor())
    mscoco_loader = DataLoader(
        mscoco_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Process MSCOCO and update the same global means
    global_running_means = process_dataset(mscoco_loader, model, global_running_means, dataset_name="MSCOCO")

    # Save combined global means
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "global_means.npz"),
        hand_feat=global_running_means['hand_feat'],
        body_feat=global_running_means['body_feat'],
        total_samples=global_running_means['total_samples']
    )