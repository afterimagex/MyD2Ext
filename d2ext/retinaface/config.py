# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_retinaface_config(cfg):
    """
    Add config for RetinaFace.
    """
    # ---------------------------------------------------------------------------- #
    # MobileNets
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.MNET = CN(new_allowed=True)

    # Output features
    cfg.MODEL.MNET.OUT_FEATURES = ['mob3', 'mob4', 'mob5']
    # Width mult
    cfg.MODEL.MNET.WIDTH_MULT = 1.0

    # ---------------------------------------------------------------------------- #
    # RetinaFace Head
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RETINAFACE = CN(new_allowed=True)

    # IoU overlap ratio [bg, fg] for labeling anchors.
    # Anchors with < bg are labeled negative (0)
    # Anchors  with >= bg and < fg are ignored (-1)
    # Anchors with >= fg are labeled positive (1)
    cfg.MODEL.RETINAFACE.IOU_THRESHOLDS = [0.2, 0.35]  # [0.4, 0.5]
    cfg.MODEL.RETINAFACE.IOU_LABELS = [0, -1, 1]

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    cfg.MODEL.RETINAFACE.PRIOR_PROB = 0.01

    # Inference cls score threshold, only anchors with score > INFERENCE_TH are
    # considered for inference (to improve speed)
    cfg.MODEL.RETINAFACE.SCORE_THRESH_TEST = 0.2  # 0.02
    # Widerface dense faces
    cfg.MODEL.RETINAFACE.TOPK_CANDIDATES_TEST = 2000
    cfg.MODEL.RETINAFACE.NMS_THRESH_TEST = 0.4

    # Weights on (dx, dy, dw, dh) for normalizing RetinaFace anchor regression targets
    cfg.MODEL.RETINAFACE.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    # Weights on (dx1, dy1, ..., dx5, dy5) for normalizing RetinaFace landmark regression targets
    cfg.MODEL.RETINAFACE.LANDMARK_REG_WEIGHTS = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)

    # Loss parameters
    cfg.MODEL.RETINAFACE.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.RETINAFACE.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.RETINAFACE.SMOOTH_L1_LOSS_BETA = 1.0  # 0.1
    cfg.MODEL.RETINAFACE.LOC_WEIGHT = 2.0
