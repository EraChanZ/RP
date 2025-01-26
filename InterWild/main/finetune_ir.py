import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import warnings
import random
import time
import torch.backends.cudnn as cudnn

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Import your model architecture
from InterWild.main.model_new_coral import get_model

# Assume you have datasets for source (with labels) and target (unlabeled)
from your_project.datasets import SourceKeypointDataset, InfraredDataset

# Import necessary modules for RegDA losses
from tllib.alignment.regda import RegressionDisparity
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric.keypoint_detection import accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune model using RegDA')
    parser.add_argument('--source_data_path', type=str, required=True,
                        help='Path to the source domain dataset')
    parser.add_argument('--target_data_path', type=str, required=True,
                        help='Path to the target domain dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the pretrained model checkpoint on source domain')
    parser.add_argument('--save_path', type=str, default='finetuned_model.pth',
                        help='Path to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr_f', type=float, default=0.001,
                        help='Learning rate for feature extractor')
    parser.add_argument('--lr_h', type=float, default=0.001,
                        help='Learning rate for main head')
    parser.add_argument('--lr_h_adv', type=float, default=0.001,
                        help='Learning rate for adversarial head')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--trade_off', type=float, default=1.0,
                        help='Trade-off parameter for the regression disparity loss')
    parser.add_argument('--margin', type=float, default=4.0,
                        help='Margin parameter for regression disparity loss')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='Print frequency')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('Using deterministic seed may slow down training!')

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data loaders for source and target domains
    # Replace with your actual dataset implementations and transformations
    source_dataset = SourceKeypointDataset(root=args.source_data_path, split='train')
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)

    target_dataset = InfraredDataset(root=args.target_data_path, split='train')
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    # Create model with main and adversarial heads
    model = get_model(mode='train')
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_data, strict=False)
    model = model.to(device)

    # Optionally freeze layers
    # for param in model.body_backbone.parameters():
    #     param.requires_grad = False

    # Define loss functions
    criterion = nn.MSELoss()

    # Regression Disparity loss for domain adaptation
    regression_disparity = RegressionDisparity(margin=args.margin)

    # Separate parameters for different optimizers
    feature_extractor_params = []
    main_head_params = []
    adv_head_params = []

    for name, param in model.named_parameters():
        if 'hand_position_net_adv' in name:
            adv_head_params.append(param)
        elif 'hand_position_net' in name:
            main_head_params.append(param)
        else:
            feature_extractor_params.append(param)

    # Define optimizers
    optimizer_f = SGD(feature_extractor_params, lr=args.lr_f, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_h = SGD(main_head_params, lr=args.lr_h, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_h_adv = SGD(adv_head_params, lr=args.lr_h_adv, momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate schedulers
    scheduler_f = LambdaLR(optimizer_f, lr_lambda=lambda epoch: 1.0)
    scheduler_h = LambdaLR(optimizer_h, lr_lambda=lambda epoch: 1.0)
    scheduler_h_adv = LambdaLR(optimizer_h_adv, lr_lambda=lambda epoch: 1.0)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()

        for i in range(min(len(source_loader), len(target_loader))):
            try:
                source_data = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_data = next(source_iter)

            try:
                target_data = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)

            data_time.update(time.time() - end)

            # Unpack source data
            inputs_s = source_data['img'].to(device)
            targets_s = source_data['joint_img'].to(device)
            meta_info_s = source_data['meta_info']
            # Include other necessary meta information from your dataset

            # Unpack target data
            inputs_t = target_data['img'].to(device)

            # -------------------
            # Step A: Train on source domain
            # -------------------
            # Zero gradients
            optimizer_f.zero_grad()
            optimizer_h.zero_grad()
            optimizer_h_adv.zero_grad()

            # Forward pass
            outputs_s_main, outputs_s_adv = model(inputs_s, targets_s, meta_info_s, mode='train')

            # Supervised loss on source domain (main head)
            loss_s = criterion(outputs_s_main, targets_s)

            # Consistency loss on source domain between main and adversarial heads (optional)
            # loss_consistency_s = criterion(outputs_s_main, outputs_s_adv)

            # Total loss on source domain
            total_loss_s = loss_s  # + loss_consistency_s * args.consistency_trade_off

            # Backward and optimize
            total_loss_s.backward()
            optimizer_f.step()
            optimizer_h.step()
            optimizer_h_adv.step()

            # -------------------
            # Step B: Maximize regression disparity on target domain (update adv head)
            # -------------------
            optimizer_h_adv.zero_grad()

            # Forward pass
            outputs_t_main, outputs_t_adv = model(inputs_t, targets=None, meta_info=None, mode='train')

            # Compute regression disparity loss (maximize)
            loss_t_adv = args.trade_off * regression_disparity(outputs_t_main, outputs_t_adv, mode='max')

            # Backward and optimize
            loss_t_adv.backward()
            optimizer_h_adv.step()

            # -------------------
            # Step C: Minimize regression disparity on target domain (update feature extractor)
            # -------------------
            optimizer_f.zero_grad()

            # Forward pass
            outputs_t_main, outputs_t_adv = model(inputs_t, targets=None, meta_info=None, mode='train')

            # Compute regression disparity loss (minimize)
            loss_t_f = args.trade_off * regression_disparity(outputs_t_main, outputs_t_adv, mode='min')

            # Backward and optimize
            loss_t_f.backward()
            optimizer_f.step()

            # Update meters
            losses_s.update(total_loss_s.item(), inputs_s.size(0))
            losses_t.update((loss_t_adv + loss_t_f).item(), inputs_t.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], '
                      f'Iter [{i}/{min(len(source_loader), len(target_loader))}], '
                      f'Loss_s: {losses_s.avg:.4f}, '
                      f'Loss_t: {losses_t.avg:.4f}, '
                      f'Data Time: {data_time.avg:.3f}, '
                      f'Batch Time: {batch_time.avg:.3f}')

        scheduler_f.step()
        scheduler_h.step()
        scheduler_h_adv.step()

    # Save the fine-tuned model
    torch.save(model.state_dict(), args.save_path)
    print(f"Fine-tuning complete. Model saved to {args.save_path}")

# Utility classes for tracking time and loss
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    main()