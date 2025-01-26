import torch
from torch.utils.data import DataLoader
import os
import os.path as osp
import sys
import gc
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# Import the same dependencies as in calculate_source_domain_encodings.py
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))

from config import cfg
from model_new_ssa import get_model
from InterHand26M_eval import InterHand26M_eval
from calculate_source_domain_encodings import MSCOCODataset

def update_covariance_perHW(features, mean, prev_covs, n_prev_samples):
    """Update covariance matrices for each spatial location (H,W)"""
    n_new_samples = features.shape[0]
    n_total = n_prev_samples + n_new_samples
    
    # features is [B, C, H, W], mean is [C, H, W]
    centered_features = features - mean.unsqueeze(0)
    
    H, W = features.shape[2:]
    if prev_covs is None:
        prev_covs = {}
    
    for h in range(H):
        for w in range(W):
            # Extract features for this spatial location [B, C]
            feat_hw = centered_features[:, :, h, w]
            
            # Calculate covariance for this location
            batch_cov = torch.matmul(feat_hw.T, feat_hw) / n_total
            
            hw_key = f"{h}_{w}"
            if hw_key not in prev_covs:
                prev_covs[hw_key] = batch_cov
            else:
                prev_covs[hw_key] = (n_prev_samples/n_total) * prev_covs[hw_key] + batch_cov
    
    return prev_covs

def process_dataset_covmat(dataloader, model, means, running_covs=None, dataset_name=""):
    print(f"\nProcessing {dataset_name} dataset for covariance...")
    
    if running_covs is None:
        running_covs = {
            'body_feat': None,
            'total_samples': 0
        }
    
    for batch_idx, (inputs, targets, meta_info) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch_imgs = inputs['img'].cuda()
            outputs = model({'img': batch_imgs}, {}, meta_info, 'test', justbodyfeat=True)
            
            # Features remain as [B, C, H, W]
            body_feats = outputs['source_body_feat']
            
            running_covs['body_feat'] = update_covariance_perHW(
                body_feats, 
                means['body_feat'], 
                running_covs['body_feat'], 
                running_covs['total_samples']
            )
            
            running_covs['total_samples'] += body_feats.shape[0]
        
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx * dataloader.batch_size}/{len(dataloader.dataset)} samples")
            print(f"Memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    return running_covs

if __name__ == '__main__':
    # ... existing setup code ...
    # Constants
    BATCH_SIZE = 64
    CHUNK_SIZE = 16384 // 2
    OUTPUT_DIR = r"D:\datasets\saved_features"
    
    # Load model
    model = get_model(mode='test')
    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(r"C:\Users\vladi\RP\InterWild\demo\snapshot_6.pth")
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    # Load means without pooling
    means_data = np.load(os.path.join(OUTPUT_DIR, "global_means.npz"))
    
    # Convert means to torch on GPU (keeping spatial dimensions)
    means = {
        'body_feat': torch.from_numpy(means_data['body_feat']).cuda()
    }
    
    print("Loaded means:")
    print(f"Body feature mean shape: {means['body_feat'].shape}")
    
    # ... rest of the processing code ...
    global_running_covs = None
    
    # Process InterHand dataset in chunks
    total_ih_samples = 1361062
    num_chunks = (total_ih_samples // CHUNK_SIZE) + 1

    for offset in tqdm(range(0, total_ih_samples, CHUNK_SIZE)):
        chunk_num = offset//CHUNK_SIZE + 1
        print(f"\nProcessing InterHand chunk {chunk_num}/{num_chunks}")
        
        ih_ds = InterHand26M_eval(
            transform=transforms.ToTensor(),
            data_split='train',
            offset=offset,
            size=min(CHUNK_SIZE, total_ih_samples - offset)
        )
        
        ih_loader = DataLoader(
            ih_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        global_running_covs = process_dataset_covmat(
            ih_loader, 
            model, 
            means,
            global_running_covs, 
            dataset_name=f"InterHand chunk {chunk_num}"
        )
        
        del ih_ds, ih_loader
        gc.collect()
        torch.cuda.empty_cache()
    # Save covariance matrices with spatial information
    covmat_dict = {
        'total_samples': global_running_covs['total_samples']
    }
    
    # Add body feature covariances with spatial information
    for hw_key, cov_matrix in global_running_covs['body_feat'].items():
        h, w = map(int, hw_key.split('_'))
        covmat_dict[f'body_feat_cov_{h}_{w}'] = cov_matrix.cpu().numpy()
    
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "spatial_covmats_body.npz"),
        **covmat_dict
    )