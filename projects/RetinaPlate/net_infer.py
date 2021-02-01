#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle
import sys

import cv2
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup

sys.path.append('../..')
sys.path.append('..')

from d2ext.config.defaults import add_retinaface_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_retinaface_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.META_ARCHITECTURE = '_RetinaFace'
    cfg.MODEL.WEIGHTS = ''
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def CenterPadResize(image, bchw_shape, border_value):
    src_h, src_w = image.shape[:2]
    _, _, dst_h, dst_w = bchw_shape
    ratio = min(float(dst_h) / src_h, float(dst_w) / src_w)
    new_size = (round(src_w * ratio), round(src_h * ratio))
    dw = (dst_w - new_size[0]) / 2
    dh = (dst_h - new_size[1]) / 2
    t, b = round(dh - 0.1), round(dh + 0.1)
    l, r = round(dw - 0.1), round(dw + 0.1)
    image = cv2.resize(image, new_size, interpolation=0)
    image = cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=border_value)
    return image


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()
    model.training = False
    # image = '/media/ps/A1/XPC/data/CCPD/ccpd_rotate/1316-21_16-167&353_530&656-530&519_183&656_167&490_514&353-0_0_6_27_27_30_22-47-40.jpg'
    image = '/home/xpc/ws/MyD2Ext/projects/RetinaPlate/2.jpg'
    image = cv2.imread(image)
    image = CenterPadResize(image, (1, 3, 640, 640), 0)
    # image = cv2.resize(image, (512, 512))
    image = np.transpose(image, (2, 0, 1))

    images = [{'image': torch.from_numpy(image)}]
    images = model.preprocess_image(images)
    pred_logits, pred_anchor_deltas, pred_keypoint_deltas = model(images.tensor)
    pred_logits = [x.cpu().detach().numpy() for x in pred_logits]
    pred_anchor_deltas = [x.cpu().detach().numpy() for x in pred_anchor_deltas]
    pred_keypoint_deltas = [x.cpu().detach().numpy() for x in pred_keypoint_deltas]

    with open('pred_logits.bin', 'wb') as f:
        pickle.dump(pred_logits, f)

    with open('pred_anchor_deltas.bin', 'wb') as f:
        pickle.dump(pred_anchor_deltas, f)

    with open('pred_keypoint_deltas.bin', 'wb') as f:
        pickle.dump(pred_keypoint_deltas, f)

    with open('anchor.bin', 'wb') as f:
        anchor = model.write_priors(images.tensor, '')
        # anchor = [
        #     model.anchor_generator.generate_cell_anchors(sizes=[16, 32], aspect_ratios=[1.0]).numpy(),
        #     model.anchor_generator.generate_cell_anchors(sizes=[64, 128], aspect_ratios=[1.0]).numpy(),
        #     model.anchor_generator.generate_cell_anchors(sizes=[256, 512], aspect_ratios=[1.0]).numpy(),
        # ]
        pickle.dump(anchor, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
