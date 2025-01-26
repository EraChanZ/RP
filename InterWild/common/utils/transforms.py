# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 


import torch
import numpy as np
from config import cfg
from torch.nn import functional as F

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:,:,0] / (width-1) * 2 - 1
    y = joint_xy[:,:,1] / (height-1) * 2 - 1
    grid = torch.stack((x,y),2)[:,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,:,0] # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0,2,1).contiguous() # batch_size, joint_num, channel_dim
    return img_feat

def soft_argmax_2d(heatmap2d):
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height*width))
    heatmap2d = F.softmax(heatmap2d, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))

    accu_x = heatmap2d.sum(dim=(2))
    accu_y = heatmap2d.sum(dim=(3))

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out

def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2,3))
    accu_y = heatmap3d.sum(dim=(2,4))
    accu_z = heatmap3d.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out

def restore_restore_bbox(bbox, aspect_ratio, extension_ratio):
    """
    Inverse function of restore_bbox. Takes bbox and returns center and size.
    
    Args:
        bbox: tensor of shape (N, 4) in xyxy format
        aspect_ratio: width/height ratio (can be None)
        extension_ratio: scaling factor used in original bbox
    
    Returns:
        bbox_center: tensor of shape (N, 2) in output_body_hm_shape space
        bbox_size: tensor of shape (N, 2) in output_body_hm_shape space
    """
    # Convert to numpy for easier manipulation
    bbox = bbox.clone()  # Create a copy to avoid modifying input
    
    # First convert from input_body_shape space to output_body_hm_shape space
    bbox[:,[0,2]] = bbox[:,[0,2]] / cfg.input_body_shape[1] * cfg.output_body_hm_shape[2]
    bbox[:,[1,3]] = bbox[:,[1,3]] / cfg.input_body_shape[0] * cfg.output_body_hm_shape[1]
    
    # Convert xyxy to xywh
    w = bbox[:,2] - bbox[:,0]
    h = bbox[:,3] - bbox[:,1]
    c_x = bbox[:,0] + w/2.
    c_y = bbox[:,1] + h/2.
    
    # Reverse extension ratio
    w = w / extension_ratio
    h = h / extension_ratio
    
    # Reverse aspect ratio preservation if needed
    if aspect_ratio is not None:
        mask1 = w > (aspect_ratio * h)
        mask2 = w < (aspect_ratio * h)
        # For mask1 cases, h was modified to w/aspect_ratio
        h[mask1] = w[mask1] / aspect_ratio
        # For mask2 cases, w was modified to h*aspect_ratio
        w[mask2] = h[mask2] * aspect_ratio
    
    # Construct center and size
    bbox_center = torch.stack([c_x, c_y], dim=1)
    bbox_size = torch.stack([w, h], dim=1)
    
    return bbox_center, bbox_size

def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio):
    bbox = bbox_center.view(-1,1,2) + torch.cat((-bbox_size.view(-1,1,2)/2., bbox_size.view(-1,1,2)/2.),1) # xyxy in (cfg.output_body_hm_shape[2], cfg.output_body_hm_shape[1]) space
    bbox[:,:,0] = bbox[:,:,0] / cfg.output_body_hm_shape[2] * cfg.input_body_shape[1]
    bbox[:,:,1] = bbox[:,:,1] / cfg.output_body_hm_shape[1] * cfg.input_body_shape[0]
    bbox = bbox.view(-1,4)

    # xyxy -> xywh
    bbox[:,2] = bbox[:,2] - bbox[:,0]
    bbox[:,3] = bbox[:,3] - bbox[:,1]
    
    w = bbox[:,2]
    h = bbox[:,3]
    c_x = bbox[:,0] + w/2.
    c_y = bbox[:,1] + h/2.

    # aspect ratio preserving bbox
    if aspect_ratio is not None:
        mask1 = w > (aspect_ratio * h)
        mask2 = w < (aspect_ratio * h)
        h[mask1] = w[mask1] / aspect_ratio
        w[mask2] = h[mask2] * aspect_ratio

    bbox[:,2] = w*extension_ratio
    bbox[:,3] = h*extension_ratio
    bbox[:,0] = c_x - bbox[:,2]/2.
    bbox[:,1] = c_y - bbox[:,3]/2.
    
    # xywh -> xyxy
    bbox[:,2] = bbox[:,2] + bbox[:,0]
    bbox[:,3] = bbox[:,3] + bbox[:,1]
    return bbox
