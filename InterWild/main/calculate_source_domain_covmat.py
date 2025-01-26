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

def update_covariance(features, mean, prev_cov, n_prev_samples):
    """Update covariance matrix with globally pooled features, keeping everything on GPU"""
    n_new_samples = features.shape[0]
    n_total = n_prev_samples + n_new_samples
    
    # features is [B, C], mean is [C] - both on GPU
    centered_features = features - mean.unsqueeze(0)
    batch_cov = torch.matmul(centered_features.T, centered_features) / n_total
    
    if prev_cov is None:
        combined_cov = batch_cov
    else:
        combined_cov = (n_prev_samples/n_total) * prev_cov + batch_cov
    
    return combined_cov

def process_dataset_covmat(dataloader, model, means, running_covs=None, dataset_name=""):
    print(f"\nProcessing {dataset_name} dataset for covariance...")
    
    # Define global pooling
    global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    
    if running_covs is None:
        running_covs = {
            'hand_feat': None,
            'body_feat': None,
            'total_samples': 0
        }
    
    for batch_idx, (inputs, targets, meta_info) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch_imgs = inputs['img'].cuda()
            outputs = model({'img': batch_imgs}, {}, meta_info, 'test')
            
            # Pool features to [B, C, 1, 1]
            hand_feats = global_pool(outputs['source_hand_feat'])
            body_feats = global_pool(outputs['source_body_feat'])
            
            # Squeeze spatial dims but keep on GPU: [B, C]
            hand_feats = hand_feats.squeeze(-1).squeeze(-1)
            body_feats = body_feats.squeeze(-1).squeeze(-1)
            
            running_covs['hand_feat'] = update_covariance(
                hand_feats, 
                means['hand_feat'], 
                running_covs['hand_feat'], 
                running_covs['total_samples']
            )
            
            running_covs['body_feat'] = update_covariance(
                body_feats, 
                means['body_feat'], 
                running_covs['body_feat'], 
                running_covs['total_samples']
            )
            
            running_covs['total_samples'] += hand_feats.shape[0]
        
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx * dataloader.batch_size}/{len(dataloader.dataset)} samples")
            print(f"Memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    return running_covs

if __name__ == '__main__':
    # Constants
    BATCH_SIZE = 16
    CHUNK_SIZE = 16384 // 2
    OUTPUT_DIR = r"D:\datasets\saved_features"
    
    # Load model
    model = get_model(mode='test')
    model = torch.nn.DataParallel(model).cuda()
    ckpt = torch.load(r"C:\Users\vladi\RP\InterWild\demo\snapshot_6.pth")
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    
    # Load and pool pre-computed means, keeping them on GPU
    means_data = np.load(os.path.join(OUTPUT_DIR, "global_means.npz"))
    global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    
    # Convert means to torch on GPU
    hand_mean = torch.from_numpy(means_data['hand_feat']).cuda().unsqueeze(0)
    body_mean = torch.from_numpy(means_data['body_feat']).cuda().unsqueeze(0)
    
    means = {
        'hand_feat': global_pool(hand_mean).squeeze(-1).squeeze(-1)[0],  # Keep on GPU
        'body_feat': global_pool(body_mean).squeeze(-1).squeeze(-1)[0]   # Keep on GPU
    }
    
    print("Loaded and pooled means:")
    print(f"Hand feature mean shape: {means['hand_feat'].shape}")
    print(f"Body feature mean shape: {means['body_feat'].shape}")
    
    # Process datasets as before
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
            num_workers=4,
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
    
    # Save covariance matrices
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "global_pooled_covmats.npz"),
        hand_feat_cov=global_running_covs['hand_feat'].cpu().numpy(),
        body_feat_cov=global_running_covs['body_feat'].cpu().numpy(),
        total_samples=global_running_covs['total_samples']
    )
