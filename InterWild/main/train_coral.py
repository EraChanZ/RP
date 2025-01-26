import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import logging
import random

from config import cfg
from base import Trainer  # from InterWild/common/base.py
from model_new_coral import get_model  # from InterWild/main/model_new_coral.py

# Example data loading stubs:
# You need to define / adapt these classes for the unlabeled IR dataset and the
# labeled InterHand26M dataset (mapped to 2D).
# --------------------------------------------------------------------
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
# For AMP (mixed precision):
import torch.cuda.amp as amp
from utils.preprocessing import load_img, process_bbox, augmentation
from pycocotools.coco import COCO
import os.path as osp
from train_ssa import CustomHandLandmarksDataset
from torch.utils.data import ConcatDataset
from custom_eval_framework import calculate_coral_train_loss
# -------------------------------------------------
# 1. Create a debug-specific logger
# -------------------------------------------------
debug_logger = logging.getLogger("coral_debug")
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False  # Prevent propagation to root logger / stdout
# Log to a separate file to avoid cluttering general train logs
debug_file_handler = logging.FileHandler("coral_debug.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_logger.addHandler(debug_file_handler)

class InfraredHandDataset(Dataset):
    """
    Minimal placeholder for unlabeled IR dataset from:
    C:\\Users\\vladi\\RP\\Research\\IR_videos\\every60thframe
    Each item returns (target_inputs, target_meta_info).
    target_inputs['img'] is your preprocessed IR image.
    target_meta_info might have any needed info (camera data, etc.) or be empty.
    """
    def __init__(self, root_paths, transform=None):
        super().__init__()
        self.root_paths = root_paths
        self.transform = transform
        # Example: gather image paths
        self.img_files = {}
        for root_path in self.root_paths:
            self.img_files[root_path] = sorted(os.listdir(root_path))

    def __len__(self):
        return sum(len(files) for files in self.img_files.values())

    def __getitem__(self, idx):
        # Find which root_path contains this index
        total = 0
        target_root = None
        target_idx = None
        for root_path, files in self.img_files.items():
            if idx < total + len(files):
                target_root = root_path
                target_idx = idx - total
                break
            total += len(files)
            
        if target_root is None:
            raise IndexError(f"Index {idx} out of bounds")
            
        img_path = os.path.join(target_root, self.img_files[target_root][target_idx])
        
        # Debug image loading
        debug_logger.debug(f"Loading image from: {img_path}")
        img = load_img(img_path)
        debug_logger.debug(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")
        debug_logger.debug(f"Image min/max values: {img.min()}/{img.max()}")
        
        body_bbox = process_bbox((0, 0, 
                                  img.shape[1], 
                                  img.shape[0]), 
                                  img.shape[1], 
                                  img.shape[0], 
                                  do_sanitize=True
                                  )
        
        debug_logger.debug(f"Processed bbox: {body_bbox}")
        
        if body_bbox is None or body_bbox[2] < 5 or body_bbox[3] < 5:
            raise ValueError(f"Bad bounding box {body_bbox} for {img_path}")

        # Debug pre-augmentation state
        debug_logger.debug(f"Pre-augmentation image: dtype={img.dtype}, shape={img.shape}, range=[{img.min()}, {img.max()}]")
        
        try:
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, "train")
        except Exception as e:
            debug_logger.error(f"Augmentation failed for {img_path}")
            debug_logger.error(f"Error: {str(e)}")
            raise

        img = self.transform(img.astype(np.float32))/255.
      
        target_inputs = {'img': img}
        # No labels, so no target-specific data
        target_meta_info = {"bb2img_trans": bb2img_trans, "orig_img_path": img_path}
        return target_inputs, target_meta_info



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', required=True)
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--source_ckpt', type=str, default='', help='Path to the source-trained checkpoint to finetune')

    # ---------------------------
    # NEW ARGS for lower memory
    # ---------------------------
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU. Adjust for limited VRAM')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                        help='Accumulate gradients over multiple mini-batches'
                             ' before calling optimizer.step().')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: "cuda" or "cpu"')
    # ---------------------------

    args = parser.parse_args()

    # Handle GPU range notation
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

class CoralCombinedDataset(torch.utils.data.Dataset):
    """
    Yields (source_inputs, source_targets, source_meta_info,
            target_inputs, target_meta_info) on the fly.
    When source dataset is larger, randomly samples from it while using
    all target dataset entries in each epoch.
    """
    def __init__(self, source_dataset, target_dataset, max_pairs=None):
        super().__init__()
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.len_source = len(self.source_dataset)
        self.len_target = len(self.target_dataset)
        print("len_source", self.len_source, "len_target", self.len_target)
        # Use target length to ensure we go through all target data
        self.length = self.len_target
        # Create random indices for source dataset sampling
        self.refresh_source_indices()
        
    def refresh_source_indices(self):
        """Create new random mapping for source dataset at start of each epoch"""
        self.source_indices = torch.randperm(self.len_source)[:self.length].tolist()
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get target data sequentially
        target_data = self.target_dataset[idx]
        # Get source data using our random mapping
        source_data = self.source_dataset[self.source_indices[idx]]
        
        source_inputs, source_targets, source_meta_info = source_data
        target_inputs, target_targets, target_meta_info = target_data
        return (source_inputs, source_targets, source_meta_info,
                target_inputs, target_targets, target_meta_info)
    
class CoralTrainer(Trainer):
    """
    A domain-adaptation trainer for CORAL. Reuses base Trainer but
    overrides _make_batch_generator to produce (source_inputs, source_targets, source_meta_info, target_inputs, target_meta_info).
    Also overrides _make_model to load the specialized CORAL model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.start_epoch = 0

        # (Optional) You can control debug logging with an arg:
        # e.g., parse "--debug" to turn debug logs on/off
        self.debug = getattr(args, 'debug', True)

    def _make_batch_generator(self):
        """Create a DataLoader that yields pairs of (source, target) data."""
        self.logger.info("Creating dataset for source & target in a single batch generator...")

        # For debug logging:
        if self.debug:
            debug_logger.debug("Entered _make_batch_generator in CoralTrainer.")

        # Source dataset (MSCOCO val)
        source_dataset = MSCOCODataset(transform=transforms.ToTensor())

        # Target IR dataset (unlabeled), now with caching
        target_dataset = InfraredHandDataset(
            root_path=r'C:\Users\vladi\RP\Research\IR_videos\every60thframe',
            transform=transforms.ToTensor()
        )

        from itertools import zip_longest
        self.combined_data = []
        max_pairs = self.args.max_batches * cfg.num_gpus * cfg.train_batch_size if hasattr(self.args, 'max_batches') else None

        for i, (s_item, t_item) in enumerate(zip_longest(source_dataset, target_dataset, fillvalue=None)):
            if t_item is None or s_item is None:
                # If lengths differ and you want to stop at the shorter dataset:
                break
            if max_pairs and i >= max_pairs:
                break

            source_inputs, source_targets, source_meta_info = s_item
            target_inputs, target_meta_info = t_item
            self.combined_data.append(
                (source_inputs, source_targets, source_meta_info, target_inputs, target_meta_info)
            )

        self.itr_per_epoch = len(self.combined_data) // (cfg.num_gpus * cfg.train_batch_size)

        self.batch_generator = DataLoader(
            dataset=self.combined_data,
            batch_size=cfg.num_gpus * cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,           # Increase this if you have enough CPU cores
            pin_memory=True,                      # Speeds up host-to-GPU copies
            prefetch_factor=2,                    # How many batches loaded in advance per worker
            persistent_workers=True,              # Workers remain active between epochs
            drop_last=True
        )
        
    def _make_batch_generator2(self):
        """Creates combined source-target dataset using base trainer's source data"""
        self.logger.info("Creating dataset for source & target in a single batch generator...")
        
        # First get source dataset from base trainer
        base_trainer = Trainer()
        base_trainer._make_batch_generator()
        source_dataset = base_trainer.trainset_loader
        
        # Create target dataset
        """
        target_dataset = InfraredHandDataset(
            root_paths=[r'C:\\Users\\vladi\\RP\\Research\\IR_videos\\every60thframe',
                        r'C:\\Users\\vladi\\RP\\our_hands_dataset_labeled_previews\\IR'],
            transform=transforms.ToTensor()
        )
        """

        target_dataset1 = CustomHandLandmarksDataset(
            image_dir='C:\\Users\\vladi\\RP\\Research\\IR_videos\\every60thframe',
            annotations_path='C:\\Users\\vladi\\Desktop\\GDAnnotateNepal\\GroundingDINO\\hand_detection_results.json',
            transform=transforms.ToTensor()
        )
        target_dataset1.annotations = [{**ann, "normalized": False} for ann in target_dataset1.annotations]
        target_dataset1.image_to_annotation = {im: {**target_dataset1.image_to_annotation[im], "normalized": False} for im in target_dataset1.image_to_annotation}

        target_dataset2 = CustomHandLandmarksDataset(
            image_dir='C:\\Users\\vladi\\RP\\our_hands_dataset_labeled_previews\\IR',
            annotations_path='C:\\Users\\vladi\\RP\\our_hands_dataset_labeled_previews\\combined_FIX_IR.json',
            transform=transforms.ToTensor()
        )
        target_dataset = ConcatDataset([target_dataset1, target_dataset2])
        
        # Create combined dataset
        self.combined_dataset = CoralCombinedDataset(
            source_dataset=source_dataset,
            target_dataset=target_dataset
        )
        
        # Create the dataloader
        self.batch_generator = torch.utils.data.DataLoader(
            dataset=self.combined_dataset,
            batch_size=cfg.num_gpus * cfg.train_batch_size,
            shuffle=True,  # We can shuffle here since CoralCombinedDataset handles source sampling
            num_workers=cfg.num_thread,
            pin_memory=True,
            drop_last=True
        )

        self.itr_per_epoch = len(self.batch_generator)

    def _make_model(self):
        self.logger.info("Creating CORAL model + optimizer...")
        if self.debug:
            debug_logger.debug("Entered _make_model in CoralTrainer.")
        model = get_model(mode='train')

        # Move model to device
        if self.args.device == 'cuda':
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(self.args.device)

        # If we have a source checkpoint to finetune
        if self.args.source_ckpt and os.path.exists(self.args.source_ckpt):
            self.logger.info(f"Finetuning from checkpoint: {self.args.source_ckpt} on device: {self.args.device}")
            ckpt = torch.load(self.args.source_ckpt)
            load_info = model.load_state_dict(ckpt['network'], strict=False)
            print("Missing keys:", load_info.missing_keys)
            print("Unexpected keys:", load_info.unexpected_keys)

        # Organize parameters into two groups with different learning rates
        last_layer_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if "hand_roi_net.backbone.layer4" in name or "hand_roi_net.backbone.layer3" in name:
                other_params.append(param)
                param.requires_grad = True
                self.logger.info(f"Keeping {name} trainable")
            else:
                param.requires_grad = False

        # Build optimizer with parameter groups
        self.optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': cfg.lr},
            {'params': last_layer_params, 'lr': cfg.lr * 10}  # 10x learning rate for last layer
        ], weight_decay=0.01)

        # Rest of the method remains the same
        if cfg.continue_train:
            start_epoch, model, self.optimizer = self.load_model(model, self.optimizer)
        else:
            start_epoch = 0

        model.train()

        self.start_epoch = start_epoch
        self.model = model

    def train(self, validation_dataset_source=None, validation_dataset_target=None):
        """
        Main training loop, with validation if dataset provided
        """
        if self.debug:
            debug_logger.debug("Starting train() loop in CoralTrainer.")

        # Initialize loss log file
        loss_log_path = os.path.join(cfg.log_dir, 'loss_log.txt')

        with open(loss_log_path, 'w') as f:
            f.write('Start of training\n')

        for epoch in range(self.start_epoch, cfg.end_epoch):
            # Refresh source indices at the start of each epoch
            self.combined_dataset.refresh_source_indices()
            
            self.set_lr(epoch)
            self.logger.info(f"[Epoch {epoch}/{cfg.end_epoch}] - Starting CORAL domain adaptation...")
            self.tot_timer.tic()
            self.read_timer.tic()

            if validation_dataset_source is not None:
                self.logger.info("Running validation...")
                eval_report = validation_dataset_source.perform_evaluation(self.model)
                
                # Write validation results to loss log
                # Replace newlines with spaces in eval report string
                eval_str = str(eval_report).replace('\n', ' ')
                with open(loss_log_path, 'a') as f:
                    f.write(f'VALIDATION_SOURCE_EPOCH_{epoch}\t{eval_str}\n')
                
                self.logger.info(f"Validation source Results:\n{eval_report}")
            if validation_dataset_target is not None:
                self.logger.info("Running validation...")
                eval_report = validation_dataset_target.perform_evaluation(self.model)
                
                # Write validation results to loss log
                # Replace newlines with spaces in eval report string
                eval_str = str(eval_report).replace('\n', ' ')
                with open(loss_log_path, 'a') as f:
                    f.write(f'VALIDATION_TARGET_EPOCH_{epoch}\t{eval_str}\n')
                self.logger.info(f"Validation target Results:\n{eval_report}")
            if validation_dataset_source is not None and validation_dataset_target is not None:
                self.logger.info("Running validation...")
                eval_report = calculate_coral_train_loss(validation_dataset_source, validation_dataset_target, 16, self.model)
                # Write validation results to loss log
                # Replace newlines with spaces in eval report string
                eval_str = str(eval_report).replace('\n', ' ')
                with open(loss_log_path, 'a') as f:
                    f.write(f'VALIDATION_CORAL_EPOCH_{epoch}\t{eval_str}\n')
                self.logger.info(f"Validation coral Results:\n{eval_report}")

            # Initialize running loss
            running_loss = 0.0
            running_loss_dict = {}

            for itr, batch_data in enumerate(self.batch_generator):
                self.read_timer.toc()
                self.gpu_timer.tic()
                

                if self.debug and itr % 10 == 0:
                    debug_logger.debug(f"[Epoch {epoch} Iter {itr}] Shapes in batch_data: "
                                       f"{[d.shape if hasattr(d,'shape') else type(d) for d in batch_data]}")

                # batch_data: (source_inputs, source_targets, source_meta_info, target_inputs, target_meta_info) [B]
                (source_inputs, source_targets, source_meta_info, 
                 target_inputs, target_targets, target_meta_info) = batch_data

                # Move everything to device
                for k in source_inputs:
                    source_inputs[k] = source_inputs[k].to(self.args.device)
                for k in source_targets:
                    source_targets[k] = source_targets[k].to(self.args.device)
                for k in source_meta_info:
                    # If meta_info holds Tensors, also send to device
                    # Otherwise skip
                    if isinstance(source_meta_info[k], torch.Tensor):
                        source_meta_info[k] = source_meta_info[k].to(self.args.device)

                for k in target_inputs:
                    target_inputs[k] = target_inputs[k].to(self.args.device)
                for k in target_targets:
                    target_targets[k] = target_targets[k].to(self.args.device)
                for k in target_meta_info:
                    if isinstance(target_meta_info[k], torch.Tensor):
                        target_meta_info[k] = target_meta_info[k].to(self.args.device)

                # Zero and forward
                self.optimizer.zero_grad()
                # ... existing code ...
                combined_meta_info = {
                    **{f"source_{k}": v for k, v in source_meta_info.items()},
                    **{f"target_{k}": v for k, v in target_meta_info.items()}
                }
                loss_dict = self.model.forward(
                    source_inputs,
                    target_inputs,
                    source_targets,
                    target_targets,
                    combined_meta_info,  # Pass combined dictionary as meta_info
                    "train",  # Pass 'train' as the mode
                    only_bbox = False,
                    only_hand = True
                )
                # ... existing code ...

                if self.debug and itr % 10 == 0:
                    for k, v in loss_dict.items():
                        debug_logger.debug(f"Loss component {k}: {v.mean().item():.6f}")

                for k, loss_val in loss_dict.items():
                    if torch.isnan(loss_val).any():
                        print(f"[DEBUG] Loss {k} is NaN before backward pass.")
                    if torch.isinf(loss_val).any():
                        print(f"[DEBUG] Loss {k} is Inf before backward pass.")

                # Loss is a dict: { 'source_joint_img': ..., 'coral_body_feat': ..., etc. }
                loss_dict = {k: v.mean() for k, v in loss_dict.items()}
                total_loss = sum(loss_dict.values())

                if torch.isnan(total_loss):
                    print("[DEBUG] total_loss is NaN, skipping backward.")

                total_loss.backward()
                self.optimizer.step()

                self.gpu_timer.toc()

                # Update running loss
                running_loss += total_loss.item()
                for k, v in loss_dict.items():
                    if k not in running_loss_dict:
                        running_loss_dict[k] = 0.0
                    running_loss_dict[k] += v.item()

                # Print progress
                screen = [
                    f'Epoch {epoch}/{cfg.end_epoch} itr {itr}/{self.itr_per_epoch}:',
                    f'lr: {self.get_lr():.6f}',
                    f'speed: {self.tot_timer.average_time:.2f}({self.gpu_timer.average_time:.2f}s r{self.read_timer.average_time:.2f})s/itr',
                    f'{(self.tot_timer.average_time/3600.0)*self.itr_per_epoch:.2f}h/epoch',
                ]
                for k in loss_dict:
                    screen.append(f'{k}: {loss_dict[k].item():.4f}')
                self.logger.info(' '.join(screen))

                # Write loss to log file
        
                
                with open(loss_log_path, 'a') as f:
                    f.write(f'{epoch}\t{itr}\t{total_loss.item():.4f}\t')
                    f.write('\t'.join([f'{v.item():.4f}' for v in loss_dict.values()]) + '\n')

                self.tot_timer.toc()
                self.tot_timer.tic()
                self.read_timer.tic()

            # After each epoch, if we have a validation dataset, evaluate
            

            # Save checkpoint
            save_dict = {
                'epoch': epoch,
                'network': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            self.save_model(save_dict, epoch)


def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    # Disable cudnn benchmarking if using CPU
    if args.device == 'cpu':
        cudnn.enabled = False
    else:
        cudnn.benchmark = True

    trainer = CoralTrainer(args)
    trainer._make_batch_generator()
    trainer._make_model()
    trainer.train()

if __name__ == "__main__":
    main()