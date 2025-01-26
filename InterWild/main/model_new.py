import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.loss import CoordLoss
from nets.resnet import ResNetBackbone
from nets.module import BoxNet, HandRoI, PositionNet
from utils.transforms import restore_bbox
from config import cfg

class Model(nn.Module):
    def __init__(self, body_backbone, body_box_net, hand_roi_net, hand_position_net):
        super(Model, self).__init__()
        self.body_backbone = body_backbone
        self.body_box_net = body_box_net
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        
        self.coord_loss = CoordLoss()

        self.trainable_modules = [self.body_backbone, self.body_box_net, self.hand_roi_net, self.hand_position_net]

    def forward(self, inputs, targets, meta_info, mode):
        # body network
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape, mode='bilinear')

        body_feat = self.body_backbone(body_img)
        (rhand_bbox_center, rhand_bbox_size, 
         lhand_bbox_center, lhand_bbox_size,
         rhand_bbox_conf, lhand_bbox_conf) = self.body_box_net(body_feat)
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach()
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1]/cfg.input_hand_shape[0], 2.0).detach()
        hand_feat, orig2hand_trans, hand2orig_trans = self.hand_roi_net(inputs['img'], rhand_bbox, lhand_bbox)
        
        # hand network
        joint_img = self.hand_position_net(hand_feat)
        rhand_num, lhand_num = len(rhand_bbox), len(lhand_bbox)
        # restore flipped left hand joint coordinates
        rjoint_img = joint_img[:rhand_num,:,:]
        ljoint_img = joint_img[rhand_num:,:,:]
        ljoint_img_x = ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        ljoint_img_x = cfg.input_hand_shape[1] - 1 - ljoint_img_x
        ljoint_img_x = ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        ljoint_img = torch.cat((ljoint_img_x[:,:,None], ljoint_img[:,:,1:]),2)

        if mode == 'train':
            # loss functions
            loss = {}
            loss['rhand_bbox_center'] = torch.abs(rhand_bbox_center - targets['rhand_bbox_center']) * meta_info['rhand_bbox_valid'][:,None]
            loss['rhand_bbox_size'] = torch.abs(rhand_bbox_size - targets['rhand_bbox_size']) * meta_info['rhand_bbox_valid'][:,None]
            loss['lhand_bbox_center'] = torch.abs(lhand_bbox_center - targets['lhand_bbox_center']) * meta_info['lhand_bbox_valid'][:,None]
            loss['lhand_bbox_size'] = torch.abs(lhand_bbox_size - targets['lhand_bbox_size']) * meta_info['lhand_bbox_valid'][:,None]
            loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
            return loss
        else:
            # Transform coordinates back to original image space
            rhand_hand2orig_trans = hand2orig_trans[:rhand_num]
            lhand_hand2orig_trans = hand2orig_trans[rhand_num:]
            
            # Transform right hand coordinates
            x = rjoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
            y = rjoint_img[:,:,1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]
            xy1 = torch.stack((x, y, torch.ones_like(x)),2)
            xy = torch.bmm(rhand_hand2orig_trans, xy1.permute(0,2,1)).permute(0,2,1)
            rjoint_img = xy

            # Transform left hand coordinates
            x = ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
            y = ljoint_img[:,:,1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]
            xy1 = torch.stack((x, y, torch.ones_like(x)),2)
            xy = torch.bmm(lhand_hand2orig_trans, xy1.permute(0,2,1)).permute(0,2,1)
            ljoint_img = xy

            # test output
            out = {}
            out['img'] = inputs['img']
            out['rhand_bbox'] = restore_bbox(rhand_bbox_center, rhand_bbox_size, None, 1.0)
            out['lhand_bbox'] = restore_bbox(lhand_bbox_center, lhand_bbox_size, None, 1.0)
            out['rhand_bbox_conf'] = rhand_bbox_conf
            out['lhand_bbox_conf'] = lhand_bbox_conf
            out['rjoint_img'] = rjoint_img
            out['ljoint_img'] = ljoint_img
            
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'rhand_bbox' in targets:
                out['rhand_bbox_target'] = targets['rhand_bbox']
                out['lhand_bbox_target'] = targets['lhand_bbox']
            if 'rhand_bbox_valid' in meta_info:
                out['rhand_bbox_valid'] = meta_info['rhand_bbox_valid']
                out['lhand_bbox_valid'] = meta_info['lhand_bbox_valid']
            return out

def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
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
