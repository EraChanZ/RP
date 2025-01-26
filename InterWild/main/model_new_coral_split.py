import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.loss import CoordLoss
from nets.resnet import ResNetBackbone
from nets.module import BoxNet, HandRoI, PositionNet
from utils.transforms import restore_bbox
from config import cfg
from utils.mano import mano

class Model(nn.Module):
    def __init__(self, body_backbone, body_box_net, hand_roi_net, hand_position_net):
        super(Model, self).__init__()
        self.body_backbone = body_backbone
        self.body_box_net = body_box_net
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        
        self.coord_loss = CoordLoss()
        self.coral_loss_weight = 10.0  # Weight for CORAL loss

        self.trainable_modules = [
            self.body_backbone,
            self.body_box_net,
            self.hand_roi_net,
            self.hand_position_net,
        ]

    def CORAL(self, source, target):
        # 1. Check for NaNs and Infs
        if torch.isnan(source).any() or torch.isinf(source).any():
            raise ValueError("Source features contain NaNs or Infs")
        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError("Target features contain NaNs or Infs")

        # 2. Check dimensions
        if source.size(1) != target.size(1):
            raise ValueError(
                f"Source and target feature dimensions do not match: "
                f"source: {source.size(1)}, target: {target.size(1)}"
            )
        
        # 3. Check for sufficient samples
        min_samples = 2  # Need at least 2 for covariance
        if source.size(0) < min_samples or target.size(0) < min_samples:
            raise ValueError(
                f"Insufficient samples for covariance calculation. "
                f"Need at least {min_samples}, but got "
                f"source: {source.size(0)}, target: {target.size(0)}"
            )

        d = source.size(1)  # dim vector

        source_c = self.compute_covariance(source)
        target_c = self.compute_covariance(target)

        # 4. Check for valid covariance matrices
        if torch.isnan(source_c).any() or torch.isinf(source_c).any():
            raise ValueError("Source covariance matrix contains NaNs or Infs")
        if torch.isnan(target_c).any() or torch.isinf(target_c).any():
            raise ValueError("Target covariance matrix contains NaNs or Infs")

        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

        loss = loss / (4 * d * d)
        return loss

    def compute_covariance(self, input_data):
        """
        Compute Covariance matrix of the input data in a more numerically stable way:
        1. Subtract the column mean from input_data to get X.
        2. Then compute (X^T X) / (n - 1).
        """
        n = input_data.size(0)  # batch_size

        # Center the data by subtracting column means
        X = input_data - input_data.mean(dim=0, keepdim=True)

        # Sample covariance
        c = X.t().matmul(X) / (n - 1)
        return c

    def forward(self, source_inputs, target_inputs, source_targets, meta_info, mode):
        import math

        # 1) Check if the inputs contain NaNs
        if torch.isnan(source_inputs['img']).any():
            print("[DEBUG] Found NaNs in source_inputs['img'] before backbone")
        if torch.isinf(source_inputs['img']).any():
            print("[DEBUG] Found infinities in source_inputs['img'] before backbone")    

        # Process source domain

        
        source_body_img = F.interpolate(
            source_inputs['img'], cfg.input_body_shape, mode='bilinear'
        )
        source_body_feat = self.body_backbone(source_body_img)
        
        (
            source_rhand_bbox_center,
            source_rhand_bbox_size, 
            source_lhand_bbox_center,
            source_lhand_bbox_size,
            source_rhand_conf,
            source_lhand_conf,
        ) = self.body_box_net(source_body_feat)

        source_rhand_bbox = restore_bbox(
            source_rhand_bbox_center,
            source_rhand_bbox_size,
            cfg.input_hand_shape[1]/cfg.input_hand_shape[0],
            2.0
        ).detach()
        
        source_lhand_bbox = restore_bbox(
            source_lhand_bbox_center,
            source_lhand_bbox_size, 
            cfg.input_hand_shape[1]/cfg.input_hand_shape[0],
            2.0
        ).detach()

        source_hand_feat, source_orig2hand_trans, source_hand2orig_trans = self.hand_roi_net(
            source_inputs['img'],
            source_rhand_bbox,
            source_lhand_bbox
        )

        source_joint_img = self.hand_position_net(source_hand_feat)

        source_rhand_num, source_lhand_num = len(source_rhand_bbox), len(source_lhand_bbox)

        source_rjoint_img = source_joint_img[:source_rhand_num,:,:]
        source_ljoint_img = source_joint_img[source_rhand_num:,:,:]

        source_ljoint_img_x = source_ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        source_ljoint_img_x = cfg.input_hand_shape[1] - 1 - source_ljoint_img_x
        source_ljoint_img_x = source_ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        source_ljoint_img = torch.cat((source_ljoint_img_x[:,:,None], source_ljoint_img[:,:,1:]), 2)

        source_rhand_orig2hand_trans = source_orig2hand_trans[:source_rhand_num]
        source_lhand_orig2hand_trans = source_orig2hand_trans[source_rhand_num:]
        source_rhand_hand2orig_trans = source_hand2orig_trans[:source_rhand_num]
        source_lhand_hand2orig_trans = source_hand2orig_trans[source_rhand_num:]
        
        source_joint_img = torch.cat((source_rjoint_img, source_ljoint_img),1)

    
        target_body_img = F.interpolate(
            target_inputs['img'], cfg.input_body_shape, mode='bilinear'
        )
        target_body_feat = self.body_backbone(target_body_img)

        (
            target_rhand_bbox_center,
            target_rhand_bbox_size,
            target_lhand_bbox_center, 
            target_lhand_bbox_size,
            target_rhand_conf,
            target_lhand_conf,
        ) = self.body_box_net(target_body_feat)

        target_rhand_bbox = restore_bbox(
            target_rhand_bbox_center,
            target_rhand_bbox_size,
            cfg.input_hand_shape[1]/cfg.input_hand_shape[0],
            2.0
        ).detach()

        target_lhand_bbox = restore_bbox(
            target_lhand_bbox_center,
            target_lhand_bbox_size,
            cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 
            2.0
        ).detach()

        target_hand_feat, target_orig2hand_trans, target_hand2orig_trans = self.hand_roi_net(
            target_inputs['img'],
            target_rhand_bbox,
            target_lhand_bbox
        )

        target_joint_img = self.hand_position_net(target_hand_feat)
        target_rhand_num, target_lhand_num = len(target_rhand_bbox), len(target_lhand_bbox)

        target_rjoint_img = target_joint_img[:target_rhand_num,:,:]
        target_ljoint_img = target_joint_img[target_rhand_num:,:,:]

        target_ljoint_img_x = target_ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        target_ljoint_img_x = cfg.input_hand_shape[1] - 1 - target_ljoint_img_x
        target_ljoint_img_x = target_ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        target_ljoint_img = torch.cat((target_ljoint_img_x[:,:,None], target_ljoint_img[:,:,1:]), 2)

        target_rhand_orig2hand_trans = target_orig2hand_trans[:target_rhand_num]
        target_lhand_orig2hand_trans = target_orig2hand_trans[target_rhand_num:]
        target_rhand_hand2orig_trans = target_hand2orig_trans[:target_rhand_num]
        target_lhand_hand2orig_trans = target_hand2orig_trans[target_rhand_num:]
        
        target_joint_img = torch.cat((target_rjoint_img, target_ljoint_img),1)

        if mode == 'train':
            # Compute losses
            loss = {}

            # Supervised losses on source
            loss['source_rhand_bbox_center'] = (
                torch.abs(source_rhand_bbox_center - source_targets['rhand_bbox_center'])
                * meta_info['source_rhand_bbox_valid'][:, None]
            )
            loss['source_rhand_bbox_size'] = (
                torch.abs(source_rhand_bbox_size - source_targets['rhand_bbox_size'])
                * meta_info['source_rhand_bbox_valid'][:, None]
            )
            loss['source_lhand_bbox_center'] = (
                torch.abs(source_lhand_bbox_center - source_targets['lhand_bbox_center'])
                * meta_info['source_lhand_bbox_valid'][:, None]
            )
            loss['source_lhand_bbox_size'] = (
                torch.abs(source_lhand_bbox_size - source_targets['lhand_bbox_size'])
                * meta_info['source_lhand_bbox_valid'][:, None]
            )

            for part_name, trans in (('right', source_rhand_orig2hand_trans), ('left', source_lhand_orig2hand_trans)):
                coord_name, trunc_name = 'joint_img', 'joint_trunc'
                x = source_targets[coord_name][:,mano.th_joint_type[part_name],0]
                y = source_targets[coord_name][:,mano.th_joint_type[part_name],1]
                z = source_targets[coord_name][:,mano.th_joint_type[part_name],2]
                trunc = meta_info["source_" + trunc_name][:,mano.th_joint_type[part_name],0]

                x = x / cfg.output_body_hm_shape[2] * cfg.input_img_shape[1]
                y = y / cfg.output_body_hm_shape[1] * cfg.input_img_shape[0]
                xy1 = torch.stack((x,y,torch.ones_like(x)),2)
                xy = torch.bmm(trans, xy1.permute(0,2,1)).permute(0,2,1)

                x, y = xy[:,:,0], xy[:,:,1]
                x = x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
                y = y / cfg.input_hand_shape[0] * cfg.output_hand_hm_shape[1]
                z = z / cfg.output_body_hm_shape[0] * cfg.output_hand_hm_shape[0]
                trunc *= ((x >= 0) * (x < cfg.output_hand_hm_shape[2]) * (y >= 0) * (y < cfg.output_hand_hm_shape[1]))

                coord = torch.stack((x,y,z),2)
                trunc = trunc[:,:,None]
                source_targets[coord_name] = torch.cat((source_targets[coord_name][:,:mano.th_joint_type[part_name][0],:], coord, source_targets[coord_name][:,mano.th_joint_type[part_name][-1]+1:,:]),1)
                meta_info["source_" + trunc_name] = torch.cat((meta_info["source_" + trunc_name][:,:mano.th_joint_type[part_name][0],:], trunc, meta_info["source_" + trunc_name][:,mano.th_joint_type[part_name][-1]+1:,:]),1)

            # ------------------------------------------------------------------------------
            #  Spatial CORAL for body features
            # ------------------------------------------------------------------------------
            body_coral_loss = 0.0
            _, c_b, h_b, w_b = source_body_feat.shape
            for i in range(h_b):
                for j in range(w_b):
                    # Each is [batch_size, channels]
                    source_feat_ij = source_body_feat[:, :, i, j]
                    target_feat_ij = target_body_feat[:, :, i, j]
                    # Compute CORAL loss for spatial location (i,j)
                    coral_loss_ij = self.CORAL(source_feat_ij, target_feat_ij)
                    body_coral_loss += coral_loss_ij

            # Average over all spatial locations
            body_coral_loss /= (h_b * w_b)
            loss['coral_body_feat'] = body_coral_loss * self.coral_loss_weight

            # ------------------------------------------------------------------------------
            #  Spatial CORAL for hand features
            # ------------------------------------------------------------------------------
            hand_coral_loss = 0.0
            _, c_h, h_h, w_h = source_hand_feat.shape
            for i in range(h_h):
                for j in range(w_h):
                    # Each is [batch_size, channels]
                    source_feat_ij = source_hand_feat[:, :, i, j]
                    target_feat_ij = target_hand_feat[:, :, i, j]
                    # Compute CORAL loss for spatial location (i,j)
                    coral_loss_ij = self.CORAL(source_feat_ij, target_feat_ij)
                    hand_coral_loss += coral_loss_ij

            # Average over all spatial locations
            hand_coral_loss /= (h_h * w_h)
            loss['coral_hand_feat'] = hand_coral_loss * self.coral_loss_weight

            loss['source_joint_img'] = self.coord_loss(source_joint_img, source_targets['joint_img'], meta_info['source_joint_trunc'], meta_info['source_is_3D'])

            return loss

        else:

            for part_name, source_trans, target_trans in (('right', source_rhand_hand2orig_trans, target_rhand_hand2orig_trans), ('left', source_lhand_hand2orig_trans, target_lhand_hand2orig_trans)):
                source_x = source_joint_img[:,mano.th_joint_type[part_name],0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
                source_y = source_joint_img[:,mano.th_joint_type[part_name],1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]

                target_x = target_joint_img[:,mano.th_joint_type[part_name],0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
                target_y = target_joint_img[:,mano.th_joint_type[part_name],1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]
                
                source_xy1 = torch.stack((source_x, source_y, torch.ones_like(source_x)),2)
                target_xy1 = torch.stack((target_x, target_y, torch.ones_like(target_x)),2)

                source_xy = torch.bmm(source_trans, source_xy1.permute(0,2,1)).permute(0,2,1)
                target_xy = torch.bmm(target_trans, target_xy1.permute(0,2,1)).permute(0,2,1)

                source_joint_img[:,mano.th_joint_type[part_name],0] = source_xy[:,:,0]
                source_joint_img[:,mano.th_joint_type[part_name],1] = source_xy[:,:,1]

                target_joint_img[:,mano.th_joint_type[part_name],0] = target_xy[:,:,0]
                target_joint_img[:,mano.th_joint_type[part_name],1] = target_xy[:,:,1]

            # Return predictions for both domains
            out = {}
            
            # Source domain outputs
            out['source_img'] = source_inputs['img']
            out['source_rhand_bbox'] = restore_bbox(source_rhand_bbox_center, source_rhand_bbox_size, None, 1.0)
            out['source_lhand_bbox'] = restore_bbox(source_lhand_bbox_center, source_lhand_bbox_size, None, 1.0)
            out['source_rhand_bbox_conf'] = source_rhand_conf
            out['source_lhand_bbox_conf'] = source_lhand_conf
            out['source_rjoint_img'] = source_joint_img[:,mano.th_joint_type['right'],:]
            out['source_ljoint_img'] = source_joint_img[:,mano.th_joint_type['left'],:]

            # Target domain outputs  
            out['target_img'] = target_inputs['img']
            out['target_rhand_bbox'] = restore_bbox(target_rhand_bbox_center, target_rhand_bbox_size, None, 1.0)
            out['target_lhand_bbox'] = restore_bbox(target_lhand_bbox_center, target_lhand_bbox_size, None, 1.0)
            out['target_rhand_bbox_conf'] = target_rhand_conf
            out['target_lhand_bbox_conf'] = target_lhand_conf
            out['target_rjoint_img'] = target_joint_img[:,mano.th_joint_type['right'],:]
            out['target_ljoint_img'] = target_joint_img[:,mano.th_joint_type['left'],:]

            # Add metadata if available
            if 'source_bb2img_trans' in meta_info:
                out['source_bb2img_trans'] = meta_info['source_bb2img_trans']
                out['target_bb2img_trans'] = meta_info['target_bb2img_trans']
            if 'source_rhand_bbox' in meta_info:
                out['source_rhand_bbox_target'] = meta_info['source_rhand_bbox']
                out['source_lhand_bbox_target'] = meta_info['source_lhand_bbox']
                out['target_rhand_bbox_target'] = meta_info['target_rhand_bbox']
                out['target_lhand_bbox_target'] = meta_info['target_lhand_bbox']
            if 'source_rhand_bbox_valid' in meta_info:
                out['source_rhand_bbox_valid'] = meta_info['source_rhand_bbox_valid']
                out['source_lhand_bbox_valid'] = meta_info['source_lhand_bbox_valid']
                out['target_rhand_bbox_valid'] = meta_info['target_rhand_bbox_valid']
                out['target_lhand_bbox_valid'] = meta_info['target_lhand_bbox_valid']

            return out


def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass

def get_model(mode):
    # Create backbone networks
    body_backbone = ResNetBackbone(cfg.body_resnet_type)
    body_box_net = BoxNet()

    # Create hand networks
    hand_backbone = ResNetBackbone(cfg.hand_resnet_type)
    hand_roi_net = HandRoI(hand_backbone)
    hand_position_net = PositionNet()

    if mode == 'train':
        # Initialize weights
        body_backbone.init_weights()
        body_box_net.apply(init_weights)
        
        hand_backbone.init_weights()
        hand_roi_net.apply(init_weights)
        hand_position_net.apply(init_weights)

    model = Model(body_backbone, body_box_net, hand_roi_net, hand_position_net)
    return model

"""
model will now receive top K eigenvectors and corresponding eigenvectors of body features covariance matrix computed on source domain. You need to apply 1x1 average adaptive pooling to body features, and the compute loss described in the screenshot based on given eigenvalues, and then add it to total loss
"""