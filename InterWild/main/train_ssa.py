import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import logging
from itertools import product
import shutil

from config import cfg
from base import Trainer
from model_new_ssa import get_model

# For AMP (mixed precision):
import torch.cuda.amp as amp
from utils.preprocessing import load_img, process_bbox, augmentation
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# -------------------------------------------------
# Debug logger setup
# -------------------------------------------------
debug_logger = logging.getLogger("ssa_debug")
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False
debug_file_handler = logging.FileHandler("ssa_debug.log")
debug_file_handler.setLevel(logging.DEBUG)
debug_logger.addHandler(debug_file_handler)

# Reuse the InfraredHandDataset from train_coral.py
#from train_coral import InfraredHandDataset


from custom_eval_framework import HandLandmarksDataset
from utils.preprocessing import sanitize_bbox
from torch.utils.data import ConcatDataset
from utils.preprocessing import generate_patch_image

class CustomHandLandmarksDataset(HandLandmarksDataset):
    def __init__(self, image_dir: str, annotations_path: str, transform=None):
        super().__init__(image_dir, annotations_path, transform)
    def process_hand_bbox(self, bbox, do_flip, img_shape, img2bb_trans):
        if bbox is None:
            bbox = np.array([0,0,1,1], dtype=np.float32).reshape(2,2) # dummy value
            bbox_valid = float(False) # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2,2) 

            # flip augmentation
            if do_flip:
                bbox[:,0] = img_shape[1] - bbox[:,0] - 1
                bbox[0,0], bbox[1,0] = bbox[1,0].copy(), bbox[0,0].copy() # xmin <-> xmax swap
            
            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4,2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:,:1])),1) 
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            bbox[:,0] = bbox[:,0] / cfg.input_img_shape[1] * cfg.output_body_hm_shape[2]
            bbox[:,1] = bbox[:,1] / cfg.input_img_shape[0] * cfg.output_body_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:,0]); xmax = np.max(bbox[:,0]);
            ymin = np.min(bbox[:,1]); ymax = np.max(bbox[:,1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            
            bbox_valid = float(True)
            bbox = bbox.reshape(2,2)

        return bbox, bbox_valid
    def __getitem__(self, idx):
        print("Entered __getitem__")
        _, ann = super().__getitem__(idx)

        if ann.normalized:
            ann.denormalize()

        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        img = load_img(image_path)

        lhand_bbox = np.array(ann.get_lhand_bbox())
        rhand_bbox = np.array(ann.get_rhand_bbox())
        print(lhand_bbox, rhand_bbox)
        lhand_bbox[2:] -= lhand_bbox[:2] # xyxy -> xywh
        rhand_bbox[2:] -= rhand_bbox[:2] # xyxy -> xywh
        lhand_bbox = sanitize_bbox(lhand_bbox, img.shape[1], img.shape[0])
        rhand_bbox = sanitize_bbox(rhand_bbox, img.shape[1], img.shape[0])
        rhand_bbox[2:] += rhand_bbox[:2] # xywh -> xyxy
        lhand_bbox[2:] += lhand_bbox[:2] # xywh -> xyxy

        img_shape = img.shape
        body_bbox = process_bbox((0, 0, 
                                  img.shape[1], 
                                  img.shape[0]), 
                                  img.shape[1], 
                                  img.shape[0], 
                                  do_sanitize=True
                                  )
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, body_bbox, "train")
        img = self.transform(img.astype(np.float32))/255.

        lhand_bbox, lhand_bbox_valid = self.process_hand_bbox(lhand_bbox, do_flip, img_shape, img2bb_trans)
        rhand_bbox, rhand_bbox_valid = self.process_hand_bbox(rhand_bbox, do_flip, img_shape, img2bb_trans)
        if do_flip:
            lhand_bbox, rhand_bbox = rhand_bbox, lhand_bbox
            lhand_bbox_valid, rhand_bbox_valid = rhand_bbox_valid, lhand_bbox_valid

        lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1])/2.; rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1])/2.; 
        lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]; rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0];
       

        return {
            "img": img
        }, {
            "rhand_bbox_center": rhand_bbox_center,
            "rhand_bbox_size": rhand_bbox_size,
            "lhand_bbox_center": lhand_bbox_center,
            "lhand_bbox_size": lhand_bbox_size
        }, {
            "bb2img_trans": bb2img_trans
        }

    def perform_evaluation(self, model, transform=None, model_type="coral", adaptive_threshold=True):
        from utils.preprocessing import load_img, process_bbox, generate_patch_image
        from config import cfg
        from custom_eval_framework import HandLandmarks, calculate_pck, calculate_iou, create_hand_landmarks_from_model_output
        model.eval()
        predictions = []
        annotations = []
        
        for img_path in self.image_files:
            image_path = os.path.join(self.image_dir, img_path)
            original_img = load_img(image_path)
            img_height, img_width = original_img.shape[:2]
            
            # Get annotation and process bboxes
            annotation = self.image_to_annotation[img_path]
            annotation = HandLandmarks(annotation, width=img_width, height=img_height)
            if annotation.normalized:
                annotation.denormalize()
            # Get hand bboxes
            lhand_bbox = np.array(annotation.get_lhand_bbox())
            rhand_bbox = np.array(annotation.get_rhand_bbox())
            
            # Convert to xywh format for processing
            lhand_bbox[2:] -= lhand_bbox[:2]  # xyxy -> xywh
            rhand_bbox[2:] -= rhand_bbox[:2]  # xyxy -> xywh
            
            # Sanitize bboxes
            lhand_bbox = sanitize_bbox(lhand_bbox, img_width, img_height)
            rhand_bbox = sanitize_bbox(rhand_bbox, img_width, img_height)
            
            # Convert back to xyxy format
            lhand_bbox[2:] += lhand_bbox[:2]  # xywh -> xyxy
            rhand_bbox[2:] += rhand_bbox[:2]  # xywh -> xyxy

            # Process full image bbox
            bbox = [0, 0, img_width, img_height]
            bbox = process_bbox(bbox, img_width, img_height)
            
            # Generate image patch
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            
            # Process hand bboxes
            lhand_bbox, lhand_bbox_valid = self.process_hand_bbox(lhand_bbox, False, original_img.shape, img2bb_trans)
            rhand_bbox, rhand_bbox_valid = self.process_hand_bbox(rhand_bbox, False, original_img.shape, img2bb_trans)
            
            # Calculate bbox centers and sizes
            lhand_bbox_center = (lhand_bbox[0] + lhand_bbox[1])/2.
            rhand_bbox_center = (rhand_bbox[0] + rhand_bbox[1])/2.
            lhand_bbox_size = lhand_bbox[1] - lhand_bbox[0]
            rhand_bbox_size = rhand_bbox[1] - rhand_bbox[0]

            # Transform and prepare image
            transform = transforms.ToTensor()
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            
            # Prepare inputs and targets
            inputs = {'img': img}
            targets = {
                'rhand_bbox_center': torch.tensor(rhand_bbox_center).cuda()[None,:],
                'rhand_bbox_size': torch.tensor(rhand_bbox_size).cuda()[None,:],
                'lhand_bbox_center': torch.tensor(lhand_bbox_center).cuda()[None,:],
                'lhand_bbox_size': torch.tensor(lhand_bbox_size).cuda()[None,:]
            }
            meta_info = {'source_bb2img_trans': torch.tensor(bb2img_trans).cuda()[None,:,:], 'target_bb2img_trans': torch.tensor(bb2img_trans).cuda()[None,:,:]}

            # Model inference
            with torch.no_grad():
                if model_type == "coral":
                    out = model(inputs, {'img': img.clone()}, targets, {}, meta_info, 'test')
                elif model_type == "ssa":
                    out = model(inputs, targets, meta_info, 'test', only_hand=False,  only_bbox=False)
                elif model_type == "classic":
                    out = model(inputs, targets, meta_info, 'test')

            prediction = create_hand_landmarks_from_model_output(out, bb2img_trans, img_width, img_height, annotation.image_name)
            predictions.append(prediction)
            annotations.append(annotation)

        # Calculate metrics
        if adaptive_threshold:
            report = calculate_pck(predictions, annotations, 2, adaptive_threshold=True, optimal_lr=True)
        else:
            report = calculate_pck(predictions, annotations, 0.05, adaptive_threshold=False, optimal_lr=True)
        report = calculate_iou(predictions, annotations, width=img_width, height=img_height, prev_report=report)
        return report

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', required=True)
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--source_ckpt', type=str, default='', 
                       help='Path to the source-trained checkpoint to finetune')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                       help='Accumulate gradients over multiple mini-batches')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--hyperparam_search', dest='hyperparam_search', action='store_true',
                       help='Perform hyperparameter search')
    args = parser.parse_args()

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

class SSATrainer(Trainer):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.start_epoch = 0
        self.debug = getattr(args, 'debug', True)

    def _make_batch_generator(self):
        self.logger.info("Creating dataset for target domain...")
        
        if self.debug:
            debug_logger.debug("Entered _make_batch_generator in SSATrainer.")

        # Target IR dataset

        """
        target_dataset = CustomHandLandmarksDataset(
            image_dir='C:\\Users\\vladi\\RP\\our_hands_dataset_labeled_previews\\IR',
            annotations_path='C:\\Users\\vladi\\RP\\our_hands_dataset_labeled_previews\\combined_FIX_IR.json',
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
        
        

        self.batch_generator = DataLoader(
            dataset=target_dataset,
            batch_size=cfg.num_gpus * cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )

        self.itr_per_epoch = len(target_dataset) // (cfg.num_gpus * cfg.train_batch_size)

    def _make_model(self, K=100):
        self.logger.info("Creating SSA model + optimizer...")
        if self.debug:
            debug_logger.debug("Entered _make_model in SSATrainer.")

        save_dir = "D:\\datasets\\saved_features"
        
        # Load body and hand components
        body_eigenvals = np.load(f"{save_dir}\\body_eigenvals.npy")
        body_eigenvecs = np.load(f"{save_dir}\\body_eigenvecs.npy")
        hand_eigenvals = np.load(f"{save_dir}\\hand_eigenvals.npy")
        hand_eigenvecs = np.load(f"{save_dir}\\hand_eigenvecs.npy")
        
        # Load means
        means_raw = np.load(f"{save_dir}\\global_means.npz")
        means = {key: means_raw[key] for key in means_raw.files}
        
        # Convert numpy arrays to torch tensors and move to correct device
        body_eigvecs = torch.from_numpy(body_eigenvecs).float().to(self.args.device)
        body_eigvals = torch.from_numpy(body_eigenvals).float().to(self.args.device)
        hand_eigvecs = torch.from_numpy(hand_eigenvecs).float().to(self.args.device)
        hand_eigvals = torch.from_numpy(hand_eigenvals).float().to(self.args.device)

        body_mu = torch.from_numpy(means['body_feat']).float().to(self.args.device)
        hand_mu = torch.from_numpy(means['hand_feat']).float().to(self.args.device)

        # Get top K eigenvectors and eigenvalues for both body and hand
        body_topk_eigvecs = body_eigvecs[:,:K]  # shape (K, D)
        body_topk_eigvals = body_eigvals[:K]     # shape (K,)
        hand_topk_eigvecs = hand_eigvecs[:,:K]   # shape (K, D)
        hand_topk_eigvals = hand_eigvals[:K]     # shape (K,)

        eigen_body_raw = np.load(f"{save_dir}\\spatial_eigenvalues.npz", allow_pickle=True)
        eigen_body = {
            key: {
                key2: (
                    torch.from_numpy(eigen_body_raw[key].item()[key2][:,:K]).float().to(self.args.device)
                    if key2 == "eigenvecs" 
                    else torch.from_numpy(eigen_body_raw[key].item()[key2][:K]).float().to(self.args.device)
                )
                for key2 in eigen_body_raw[key].item().keys()
            }
            for key in eigen_body_raw.files
        }

        model = get_model(
            mode='train', 
            body_topk_eigvecs=None,
            body_topk_eigvals=None,
            body_mu_s=None,
            hand_topk_eigvecs=hand_topk_eigvecs,
            hand_topk_eigvals=hand_topk_eigvals,
            hand_mu_s=hand_mu,
            body_spatial_eigendata=None,
            hand_spatial_eigendata=None
        )

        if self.args.device == 'cuda':
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(self.args.device)

        if self.args.source_ckpt and os.path.exists(self.args.source_ckpt):
            self.logger.info(f"Loading source checkpoint: {self.args.source_ckpt}")
            ckpt = torch.load(self.args.source_ckpt)
            load_info = model.load_state_dict(ckpt['network'], strict=False)
            print("Missing keys:", load_info.missing_keys)
            print("Unexpected keys:", load_info.unexpected_keys)

        # Freeze all layers except batch norm layers
        layerstokeep = []
        if  model.module.body_topk_eigvecs is not None:
            layerstokeep.append("body_backbone")
        if model.module.hand_topk_eigvecs is not None:
            layerstokeep.append("hand_roi_net")

        for name, param in model.named_parameters():
            if any(layer in name for layer in layerstokeep) and "bn" in name:
                param.requires_grad = True
                self.logger.info(f"Keeping {name} trainable")
            else:
                param.requires_grad = False

        train_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            train_params,
            lr=cfg.lr ,  # Lower learning rate for fine-tuning
            weight_decay=0.01,  # Stronger weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
        

        if cfg.continue_train:
            start_epoch, model, self.optimizer = self.load_model(model, self.optimizer)
        else:
            start_epoch = 0

        model.train()
        self.start_epoch = start_epoch
        self.model = model

    def train(self, validation_dataset_source=None, validation_dataset_target=None):
        if self.debug:
            debug_logger.debug("Starting train() loop in SSATrainer.")
        loss_log_path = os.path.join(cfg.log_dir, 'ssa_loss_log.txt')
        with open(loss_log_path, 'w') as f:
            f.write('Start of training\n')

        for epoch in range(self.start_epoch, cfg.end_epoch):
            self.set_lr(epoch)
            self.logger.info(f"[Epoch {epoch}/{cfg.end_epoch}] - Training SSA...")
            self.tot_timer.tic()
            self.read_timer.tic()
            if validation_dataset_source is not None:
                self.logger.info("Running validation...")
                eval_report = validation_dataset_source.perform_evaluation(self.model, model_type="ssa")
                
                eval_str = str(eval_report).replace('\n', ' ')
                with open(loss_log_path, 'a') as f:
                    f.write(f'VALIDATION_SOURCE_EPOCH_{epoch}\t{eval_str}\n')
                
                self.logger.info(f"Validation source Results:\n{eval_report}")
            if validation_dataset_target is not None:
                self.logger.info("Running validation...")
                eval_report = validation_dataset_target.perform_evaluation(self.model, model_type="ssa")
                
                eval_str = str(eval_report).replace('\n', ' ')
                with open(loss_log_path, 'a') as f:
                    f.write(f'VALIDATION_TARGET_EPOCH_{epoch}\t{eval_str}\n')
                self.logger.info(f"Validation target Results:\n{eval_report}")
            running_loss = 0.0

            for itr, (target_inputs, target_targets, target_meta_info) in enumerate(self.batch_generator):
                self.read_timer.toc()
                self.gpu_timer.tic()

                # Move data to device
                for k in target_inputs:
                    target_inputs[k] = target_inputs[k].to(self.args.device)
                for k in target_targets:
                    target_targets[k] = target_targets[k].to(self.args.device)
                for k in target_meta_info:
                    if isinstance(target_meta_info[k], torch.Tensor):
                        target_meta_info[k] = target_meta_info[k].to(self.args.device)

                self.optimizer.zero_grad()
                
                # Forward pass - note we pass None for source inputs/targets
                loss_dict = self.model.forward(
                    target_inputs,
                    target_targets,
                    target_meta_info,
                    "train",
                    only_hand=True
                )

                if self.debug and itr % 10 == 0:
                    for k, v in loss_dict.items():
                        debug_logger.debug(f"Loss component {k}: {v.mean().item():.6f}")

                loss_dict = {k: v.mean() for k, v in loss_dict.items()}
                total_loss = sum(loss_dict.values())

                total_loss.backward()
                self.optimizer.step()

                self.gpu_timer.toc()
                running_loss += total_loss.item()

                # Print progress
                screen = [
                    f'Epoch {epoch}/{cfg.end_epoch} itr {itr}/{self.itr_per_epoch}:',
                    f'lr: {self.get_lr():.6f}',
                    f'speed: {self.tot_timer.average_time:.2f}({self.gpu_timer.average_time:.2f}s r{self.read_timer.average_time:.2f})s/itr',
                    f'{(self.tot_timer.average_time/3600.0)*self.itr_per_epoch:.2f}h/epoch',
                ]
                screen.extend([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()])
                self.logger.info(' '.join(screen))

                with open(loss_log_path, 'a') as f:
                    f.write(f'{epoch}\t{itr}\t{total_loss.item():.4f}\t')
                    f.write('\t'.join([f'{v.item():.4f}' for v in loss_dict.values()]) + '\n')
               

                self.tot_timer.toc()
                self.tot_timer.tic()
                self.read_timer.tic()

            # Save checkpoint
            save_dict = {
                'epoch': epoch,
                'network': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            self.save_model(save_dict, epoch)

def run_hyperparameter_search(base_args):
    """
    Perform hyperparameter search for SSA model training.
    
    Args:
        base_args: Base argument namespace containing common parameters
    """

    from custom_eval_framework import HandLandmarksDataset
    val_dataset_ir = HandLandmarksDataset(image_dir=r"C:\Users\vladi\RP\our_hands_dataset_labeled_previews\IR", annotations_path=r"C:\Users\vladi\RP\our_hands_dataset_labeled_previews\combined_FIX_IR.json")
    # Define hyperparameter search space
    k_values = [5, 10, 30, 50, 70, 100, 150, 200, 300]
    lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    
    # Create base output directory for this search
    base_output_dir = os.path.join(cfg.output_dir, 'hyperparam_search')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Store original paths
    original_log_dir = cfg.log_dir
    original_model_dir = cfg.model_dir
    
    # Try all combinations
    for k, lr in product(k_values, lr_values):
        # Create unique directory for this combination
        run_name = f'K{k}_lr{lr:.0e}'
        run_dir = os.path.join(base_output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Update paths for this run
        cfg.log_dir = os.path.join(run_dir, 'log')
        cfg.model_dir = os.path.join(run_dir, 'model_dump')
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        
        # Update config for this run
        cfg.lr = lr
        cfg.end_epoch = 6  # Only run 8 epochs
        cfg.train_batch_size = 16
        
        print(f"\n{'='*50}")
        print(f"Starting run with K={k}, lr={lr}")
        print(f"{'='*50}\n")
        
        try:
            # Initialize trainer
            trainer = SSATrainer(base_args)
            trainer._make_batch_generator()
            trainer._make_model(K=k)  # Pass K parameter here
            trainer.train(validation_dataset=val_dataset_ir)
            
            # Rename snapshot files to include parameters
            for epoch in range(cfg.end_epoch):
                old_path = os.path.join(cfg.model_dir, f'snapshot_{epoch}.pth.tar')
                if os.path.exists(old_path):
                    new_path = os.path.join(cfg.model_dir, f'snapshot_{epoch}_K{k}_lr{lr:.0e}.pth.tar')
                    shutil.move(old_path, new_path)
            
        except Exception as e:
            print(f"Error during run K={k}, lr={lr}: {str(e)}")
            # Log the error
            with open(os.path.join(base_output_dir, 'errors.log'), 'a') as f:
                f.write(f"Error in run K={k}, lr={lr}: {str(e)}\n")
            continue
    
    # Restore original paths
    cfg.log_dir = original_log_dir
    cfg.model_dir = original_model_dir

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    
    if args.device == 'cpu':
        cudnn.enabled = False
    else:
        cudnn.benchmark = True

    # For regular training
    if not getattr(args, 'hyperparam_search', False):
        trainer = SSATrainer(args)
        trainer._make_batch_generator()
        trainer._make_model()
        trainer.train()
    else:
        # For hyperparameter search
        run_hyperparameter_search(args)

if __name__ == "__main__":
    main()

