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
    def __init__(
        self,
        body_backbone,
        body_box_net,
        hand_roi_net,
        hand_position_net,
        body_topk_eigvecs=None,
        body_topk_eigvals=None,
        body_mu_s=None,
        hand_topk_eigvecs=None,
        hand_topk_eigvals=None,
        hand_mu_s=None,
        body_spatial_eigendata=None,
        hand_spatial_eigendata=None
    ):
        super(Model, self).__init__()
        self.body_backbone = body_backbone
        self.body_box_net = body_box_net
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        
        self.coord_loss = CoordLoss()
        # Store the subspace alignment parameters for body
        self.body_topk_eigvecs = body_topk_eigvecs
        self.body_topk_eigvals = body_topk_eigvals
        self.body_mu_s = body_mu_s

        # Store the subspace alignment parameters for hand
        self.hand_topk_eigvecs = hand_topk_eigvecs
        self.hand_topk_eigvals = hand_topk_eigvals
        self.hand_mu_s = hand_mu_s

        self.body_spatial_eigendata = body_spatial_eigendata
        self.hand_spatial_eigendata = hand_spatial_eigendata

        self.trainable_modules = [
            self.body_backbone,
            self.body_box_net,
            self.hand_roi_net,
            self.hand_position_net,
        ]

    def compute_ssa_loss(self, features, mu_s, topk_eigvecs, topk_eigvals):
        # Helper function to compute SSA loss for any feature set
        centered = features - mu_s

        pooled_feat = F.adaptive_avg_pool2d(centered, (1,1)).view(
            centered.size(0),
            -1
        )  # z_i^t in R^D

        # Compute z_i^t - mu^s
        # self.mu_s is shape (D); broadcast subtract
            # (batch_size, D)

        # Project into the K-dimensional subspace: z̃_i^t = V^s (z_i^t - μ^s)
        # if self.topk_eigvecs is shape (K, D), then:
        z_tilde = torch.matmul(pooled_feat, topk_eigvecs)  # (batch_size, K)

        # Compute mean & var over the mini-batch in the K-dim subspace
        z_tilde_mean = z_tilde.mean(dim=0)      # μ̃^t in R^K
        z_tilde_var = z_tilde.var(dim=0, unbiased=False)  # σ̃^t2 in R^K

        # The symmetrical KL between N(0, λ^s) and N(μ̃^t, σ̃^t2) dimension-wise
        # eq(8) from the text, with alpha_d = 1 for each dimension:
        #  1/2 Σ_d [ ((μ̃_d^t)^2 + λ_d^s)/σ̃_d^t2 + ((μ̃_d^t)^2 + σ̃_d^t2)/λ_d^s - 2 ]
        # self.topk_eigvals -> λ^s in R^K
        eps = 1e-7  # small for numerical stability
        var_plus_eps = z_tilde_var + eps
        eigvals_plus_eps = topk_eigvals + eps

        term1 = (z_tilde_mean**2 + topk_eigvals) / var_plus_eps
        term2 = (z_tilde_mean**2 + var_plus_eps) / eigvals_plus_eps
        ssa_loss = 0.5 * torch.sum(term1 + term2 - 2.0)
        return ssa_loss

    def forward(self, inputs, targets, meta_info, mode, only_bbox=False, only_hand=False):
        loss = {}
        
        # Process body features if not only_hand
        if not only_hand:
            source_body_img = F.interpolate(
                inputs['img'], cfg.input_body_shape, mode='bilinear'
            )
            source_body_feat = self.body_backbone(source_body_img)
            
            if mode == 'train' and self.body_topk_eigvecs is not None:
                loss['body_ssa_loss'] = self.compute_ssa_loss(
                    source_body_feat, 
                    self.body_mu_s, 
                    self.body_topk_eigvecs, 
                    self.body_topk_eigvals
                )
                
            if mode == 'train' and self.body_spatial_eigendata is not None:
                W, H = source_body_feat.size(2), source_body_feat.size(3)   
                for i in range(W):
                    for j in range(H):
                        feat_ij = source_body_feat[:,:,i,j].view(source_body_feat.size(0), -1)
                        eigenvec_ij = self.body_spatial_eigendata[f"body_feat_cov_{i}_{j}"]["eigenvecs"]
                        eigenval_ij = self.body_spatial_eigendata[f"body_feat_cov_{i}_{j}"]["eigenvals"]
                        loss[f'body_ssa_loss_{i}_{j}'] = self.compute_ssa_loss(
                            feat_ij, 
                            None,  # No mean for spatial features
                            eigenvec_ij, 
                            eigenval_ij
                        ) * 0.01

            (source_rhand_bbox_center,
            source_rhand_bbox_size, 
            source_lhand_bbox_center,
            source_lhand_bbox_size,
            source_rhand_conf,
            source_lhand_conf,
            ) = self.body_box_net(source_body_feat)
        else:
            print("only hand")
            # Use target bbox when only processing hands
            source_rhand_bbox_center = targets["rhand_bbox_center"]
            source_rhand_bbox_size = targets["rhand_bbox_size"]
            source_lhand_bbox_center = targets["lhand_bbox_center"]
            source_lhand_bbox_size = targets["lhand_bbox_size"]
            source_rhand_conf = 1.0
            source_lhand_conf = 1.0

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

        if only_bbox:
            out = {}
            out['rhand_bbox'] = restore_bbox(source_rhand_bbox_center, source_rhand_bbox_size, None, 1.0)
            out['lhand_bbox'] = restore_bbox(source_lhand_bbox_center, source_lhand_bbox_size, None, 1.0)
            out['rhand_bbox_conf'] = source_rhand_conf
            out['lhand_bbox_conf'] = source_lhand_conf
            return out

    
        source_hand_feat, source_orig2hand_trans, source_hand2orig_trans = self.hand_roi_net(
            inputs['img'],
            source_rhand_bbox,
            source_lhand_bbox
        )

        if mode == 'train':
            if self.hand_topk_eigvecs is not None:
                loss['hand_ssa_loss'] = self.compute_ssa_loss(
                    source_hand_feat, 
                    self.hand_mu_s, 
                    self.hand_topk_eigvecs, 
                    self.hand_topk_eigvals
                )
            return loss

        # Rest of the inference code remains the same
        source_joint_img = self.hand_position_net(source_hand_feat)

        source_rhand_num, source_lhand_num = len(source_rhand_bbox), len(source_lhand_bbox)

        source_rjoint_img = source_joint_img[:source_rhand_num,:,:]
        source_ljoint_img = source_joint_img[source_rhand_num:,:,:]

        # Flip left-hand horizontally
        source_ljoint_img_x = source_ljoint_img[:,:,0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
        source_ljoint_img_x = cfg.input_hand_shape[1] - 1 - source_ljoint_img_x
        source_ljoint_img_x = source_ljoint_img_x / cfg.input_hand_shape[1] * cfg.output_hand_hm_shape[2]
        source_ljoint_img = torch.cat((source_ljoint_img_x[:,:,None], source_ljoint_img[:,:,1:]), 2)

        source_rhand_orig2hand_trans = source_orig2hand_trans[:source_rhand_num]
        source_lhand_orig2hand_trans = source_orig2hand_trans[source_rhand_num:]
        source_rhand_hand2orig_trans = source_hand2orig_trans[:source_rhand_num]
        source_lhand_hand2orig_trans = source_hand2orig_trans[source_rhand_num:]
        
        source_joint_img = torch.cat((source_rjoint_img, source_ljoint_img),1)
        # Inference / validation: map hand coordinates back to original images
        for part_name, source_trans in (
            ('right', source_rhand_hand2orig_trans),
            ('left', source_lhand_hand2orig_trans)
        ):
            source_x = source_joint_img[:,mano.th_joint_type[part_name],0] / cfg.output_hand_hm_shape[2] * cfg.input_hand_shape[1]
            source_y = source_joint_img[:,mano.th_joint_type[part_name],1] / cfg.output_hand_hm_shape[1] * cfg.input_hand_shape[0]
            
            source_xy1 = torch.stack((source_x, source_y, torch.ones_like(source_x)),2)
            source_xy = torch.bmm(source_trans, source_xy1.permute(0,2,1)).permute(0,2,1)

            source_joint_img[:,mano.th_joint_type[part_name],0] = source_xy[:,:,0]
            source_joint_img[:,mano.th_joint_type[part_name],1] = source_xy[:,:,1]

        out = {}
        out['img'] = inputs['img']
        out['rhand_bbox'] = restore_bbox(
            source_rhand_bbox_center,
            source_rhand_bbox_size,
            None, 1.0
        )
        out['lhand_bbox'] = restore_bbox(
            source_lhand_bbox_center,
            source_lhand_bbox_size,
            None, 1.0
        )
        out['rhand_bbox_conf'] = source_rhand_conf
        out['lhand_bbox_conf'] = source_lhand_conf
        out['rjoint_img'] = source_joint_img[:,mano.th_joint_type['right'],:]
        out['ljoint_img'] = source_joint_img[:,mano.th_joint_type['left'],:]
        out['source_body_feat'] = source_body_feat.detach()
        out['source_hand_feat'] = source_hand_feat.detach()

        if 'bb2img_trans' in meta_info:
            out['bb2img_trans'] = meta_info['bb2img_trans']
        if 'rhand_bbox' in meta_info:
            out['rhand_bbox_target'] = meta_info['rhand_bbox']
            out['lhand_bbox_target'] = meta_info['lhand_bbox']
        if 'rhand_bbox_valid' in meta_info:
            out['rhand_bbox_valid'] = meta_info['rhand_bbox_valid']
            out['lhand_bbox_valid'] = meta_info['lhand_bbox_valid']

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

def get_model(
    mode, 
    body_topk_eigvecs=None, 
    body_topk_eigvals=None, 
    body_mu_s=None,
    hand_topk_eigvecs=None, 
    hand_topk_eigvals=None, 
    hand_mu_s=None,
    body_spatial_eigendata=None,
    hand_spatial_eigendata=None
):
    """
    Create the model, passing in the top K eigenvectors and eigenvalues
    (and the source means) for both body and hand Subspace Alignment (SSA).
    """
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

    # Pass separate subspace parameters for body and hand to the model
    model = Model(
        body_backbone,
        body_box_net,
        hand_roi_net,
        hand_position_net,
        body_topk_eigvecs=body_topk_eigvecs,
        body_topk_eigvals=body_topk_eigvals,
        body_mu_s=body_mu_s,
        hand_topk_eigvecs=hand_topk_eigvecs,
        hand_topk_eigvals=hand_topk_eigvals,
        hand_mu_s=hand_mu_s,
        body_spatial_eigendata=body_spatial_eigendata,
        hand_spatial_eigendata=hand_spatial_eigendata
    )
    return model

"""
save_dir = "D:\\datasets\\saved_features"
        
        # Load body and hand components
        body_eigenvals = np.load(f"{save_dir}\\body_eigenvals.npy")
        body_eigenvecs = np.load(f"{save_dir}\\body_eigenvecs.npy")
        
        # Load means
        means_raw = np.load(f"{save_dir}\\global_means.npz")
        means = {key: means_raw[key] for key in means_raw.files}
        
        # Convert numpy arrays to torch tensors and move to correct device
        body_eigvecs = torch.from_numpy(body_eigenvecs).float().to(self.args.device)
        body_eigvals = torch.from_numpy(body_eigenvals).float().to(self.args.device)
    
        body_mu = torch.from_numpy(means['body_feat_mean']).float().to(self.args.device)
"""