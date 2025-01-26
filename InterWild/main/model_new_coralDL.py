import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.loss import CoordLoss
from nets.resnet import ResNetBackbone
from nets.module import BoxNet, HandRoI, PositionNet
from utils.transforms import restore_bbox
from config import cfg
from utils.mano import mano

class ModelDL(nn.Module):
    """
    This class is a modified version of the original Model in model_new_coral.py.
    Instead of using spatial pooling (SPP) directly for CORAL, we insert an
    additional learnable layer ('body_feat_reducer' and 'hand_feat_reducer')
    that reduces the dimensionality of feature maps before CORAL is applied.

    Rationale:
    1. Allows learning a more flexible, data-dependent transformation before CORAL.
    2. Minimizes dimensional mismatch if body/hand features are large (e.g. 2048-D).
    3. Lets you fine-tune the new layers from scratch while optionally keeping other
       parts of the network frozen or partially frozen.
    4. Pay attention to initialization strategies, LR scheduling, and checkpoint
       loading so that only these new layers are initialized with random weights.
    """
    def __init__(self, body_backbone, body_box_net, hand_roi_net, hand_position_net):
        super(ModelDL, self).__init__()
        self.body_backbone = body_backbone
        self.body_box_net = body_box_net
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        
        self.coord_loss = CoordLoss()
        self.coral_loss_weight = 100.0  # Weight for CORAL loss

        # NEW: simple learnable reducers to lower feature dimension before CORAL
        # (these layers will be fresh/ randomly initialized at the start of your finetune)
        # You can tweak in_channels/out_channels to match your architecture or requirements
        self.body_feat_reducer = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.hand_feat_reducer = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        # Trainable modules for convenience (includes new reducers)
        self.trainable_modules = [
            self.body_backbone,
            self.body_box_net,
            self.hand_roi_net,
            self.hand_position_net,
            self.body_feat_reducer,       # newly added
            self.hand_feat_reducer        # newly added
        ]

    def CORAL(self, source, target):
        """
        Implementation of CORAL loss.
        """
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

        # CORAL formula: sum of elementwise square diff between source & target cov
        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
        loss = loss / (4 * d * d)
        return loss

    def compute_covariance(self, input_data):
        """
        Compute covariance matrix in a numerically stable way:
        1. Subtract column mean from input_data (-> X).
        2. Then compute (X^T X) / (n - 1).
        """
        n = input_data.size(0)  # batch_size
        # Center the data
        X = input_data - input_data.mean(dim=0, keepdim=True)
        # Covariance
        c = X.t().matmul(X) / (n - 1)
        return c

    def forward(self, source_inputs, target_inputs, source_targets, meta_info, mode):
        import math

        # Quick checks
        if torch.isnan(source_inputs['img']).any():
            print("[DEBUG] Found NaNs in source_inputs['img'] before backbone")
        if torch.isinf(source_inputs['img']).any():
            print("[DEBUG] Found infinities in source_inputs['img'] before backbone")

        # ---------------------
        # Source Domain
        # ---------------------
        source_body_img = F.interpolate(source_inputs['img'], cfg.input_body_shape, mode='bilinear')
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

        # Flip left hand horizontally
        source_ljoint_img_x = source_ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        source_ljoint_img_x = cfg.input_hand_shape[1] - 1 - source_ljoint_img_x
        source_ljoint_img_x = source_ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        source_ljoint_img = torch.cat((source_ljoint_img_x[:,:,None], source_ljoint_img[:,:,1:]), 2)

        source_rhand_orig2hand_trans = source_orig2hand_trans[:source_rhand_num]
        source_lhand_orig2hand_trans = source_orig2hand_trans[source_rhand_num:]
        source_rhand_hand2orig_trans = source_hand2orig_trans[:source_rhand_num]
        source_lhand_hand2orig_trans = source_hand2orig_trans[source_rhand_num:]
        
        source_joint_img = torch.cat((source_rjoint_img, source_ljoint_img),1)

        # ---------------------
        # Target Domain
        # ---------------------
        target_body_img = F.interpolate(target_inputs['img'], cfg.input_body_shape, mode='bilinear')
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

        # Flip left hand horizontally
        target_ljoint_img_x = target_ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        target_ljoint_img_x = cfg.input_hand_shape[1] - 1 - target_ljoint_img_x
        target_ljoint_img_x = target_ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        target_ljoint_img = torch.cat((target_ljoint_img_x[:,:,None], target_ljoint_img[:,:,1:]), 2)

        target_rhand_orig2hand_trans = target_orig2hand_trans[:target_rhand_num]
        target_lhand_orig2hand_trans = target_orig2hand_trans[target_rhand_num:]
        target_rhand_hand2orig_trans = target_hand2orig_trans[:target_rhand_num]
        target_lhand_hand2orig_trans = target_hand2orig_trans[target_rhand_num:]
        
        target_joint_img = torch.cat((target_rjoint_img, target_ljoint_img),1)

        # ---------------------
        # TRAIN MODE
        # ---------------------
        if mode == 'train':
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

            # Remap GT for source joints: same as original
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
                meta_info["source_" + trunc_name] = torch.cat(
                    (
                        meta_info["source_" + trunc_name][:,:mano.th_joint_type[part_name][0],:],
                        trunc,
                        meta_info["source_" + trunc_name][:,mano.th_joint_type[part_name][-1]+1:,:]
                    ),1
                )

            # -----------------------------------------------------
            # Instead of SPP, we now use the learnable reducers
            # -----------------------------------------------------
            # 1) Body features
            source_body_feat_reduced = self.body_feat_reducer(source_body_feat)
            source_body_feat_reduced = source_body_feat_reduced.view(source_body_feat_reduced.size(0), -1)
            target_body_feat_reduced = self.body_feat_reducer(target_body_feat)
            target_body_feat_reduced = target_body_feat_reduced.view(target_body_feat_reduced.size(0), -1)

            # 2) Hand features
            source_hand_feat_reduced = self.hand_feat_reducer(source_hand_feat)
            source_hand_feat_reduced = source_hand_feat_reduced.view(source_hand_feat_reduced.size(0), -1)
            target_hand_feat_reduced = self.hand_feat_reducer(target_hand_feat)
            target_hand_feat_reduced = target_hand_feat_reduced.view(target_hand_feat_reduced.size(0), -1)

            # CORAL losses with the new reduced features
            coral_loss_body = self.CORAL(source_body_feat_reduced, target_body_feat_reduced)
            loss['coral_body_feat'] = coral_loss_body * self.coral_loss_weight

            coral_loss_hand = self.CORAL(source_hand_feat_reduced, target_hand_feat_reduced)
            loss['coral_hand_feat'] = coral_loss_hand * self.coral_loss_weight

            # Source joint supervision
            loss['source_joint_img'] = self.coord_loss(
                source_joint_img,
                source_targets['joint_img'],
                meta_info['source_joint_trunc'],
                meta_info['source_is_3D']
            )
            return loss

        else:
            # EVAL MODE
            for part_name, source_trans, target_trans in (
                ('right', source_rhand_hand2orig_trans, target_rhand_hand2orig_trans),
                ('left', source_lhand_hand2orig_trans, target_lhand_hand2orig_trans)
            ):
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

            out = {}
            # Source domain outputs
            out['source_img'] = source_inputs['img']
            out['source_rhand_bbox'] = restore_bbox(
                source_rhand_bbox_center, source_rhand_bbox_size, None, 1.0
            )
            out['source_lhand_bbox'] = restore_bbox(
                source_lhand_bbox_center, source_lhand_bbox_size, None, 1.0
            )
            out['source_rhand_bbox_conf'] = source_rhand_conf
            out['source_lhand_bbox_conf'] = source_lhand_conf
            out['source_rjoint_img'] = source_joint_img[:,mano.th_joint_type['right'],:]
            out['source_ljoint_img'] = source_joint_img[:,mano.th_joint_type['left'],:]

            # Target domain outputs  
            out['target_img'] = target_inputs['img']
            out['target_rhand_bbox'] = restore_bbox(
                target_rhand_bbox_center, target_rhand_bbox_size, None, 1.0
            )
            out['target_lhand_bbox'] = restore_bbox(
                target_lhand_bbox_center, target_lhand_bbox_size, None, 1.0
            )
            out['target_rhand_bbox_conf'] = target_rhand_conf
            out['target_lhand_bbox_conf'] = target_lhand_conf
            out['target_rjoint_img'] = target_joint_img[:,mano.th_joint_type['right'],:]
            out['target_ljoint_img'] = target_joint_img[:,mano.th_joint_type['left'],:]

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
    """
    Creates and returns the ModelDL object. This is analogous to get_model
    from the original file but references the new ModelDL class.
    
    Points to consider:
    1. If you're using a pre-trained checkpoint from the original model,
       you may want to load it in such a way that all original submodules
       (body_backbone, body_box_net, ... ) are loaded, but the new reducer
       layers (body_feat_reducer, hand_feat_reducer) are left randomly init.
    2. Possibly freeze or partially freeze the old modules if you only want
       to train these new layers for CORAL. Adjust lr parameters accordingly.
    """
    body_backbone = ResNetBackbone(cfg.body_resnet_type)
    body_box_net = BoxNet()

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

        # Initialize the reducer layers
        new_model = ModelDL(body_backbone, body_box_net, hand_roi_net, hand_position_net)
        new_model.body_feat_reducer.apply(init_weights)
        new_model.hand_feat_reducer.apply(init_weights)
        return new_model

    new_model = ModelDL(body_backbone, body_box_net, hand_roi_net, hand_position_net)
    return new_model