def load_model(model_path):
    import sys
    import os
    import os.path as osp
    import argparse
    import numpy as np
    import cv2
    import json
    import torch
    from glob import glob
    from tqdm import tqdm
    import torchvision.transforms as transforms
    from torch.nn.parallel.data_parallel import DataParallel
    import torch.backends.cudnn as cudnn

    sys.path.insert(0, osp.join('..', 'main'))
    sys.path.insert(0, osp.join('..', 'data'))
    sys.path.insert(0, osp.join('..', 'common'))
    from config import cfg
    from model import get_model
    from utils.preprocessing import load_img, process_bbox, generate_patch_image, get_iou
    from utils.vis import vis_keypoints_with_skeleton, save_obj, render_mesh_orthogonal
    from utils.mano import mano
    cfg.set_args("0")
    cudnn.benchmark = True
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_model('test')
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    return model
