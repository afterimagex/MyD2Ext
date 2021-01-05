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
    cfg.MODEL.MNET.OUT_FEATURES = ['mob3', 'mob4', 'mob5']
    cfg.MODEL.MNET.WIDTH_MULT = 1.0

    # ---------------------------------------------------------------------------- #
    # RetinaFace
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RETINAFACE = CN(new_allowed=True)
    cfg.MODEL.RETINAFACE.USE_SSH = True
    cfg.MODEL.RETINAFACE.NUM_LANDMARK = 5
    cfg.MODEL.RETINAFACE.LANDMARK_REG_WEIGHTS = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    cfg.MODEL.RETINAFACE.LOC_WEIGHT = 2.0

    # ---------------------------------------------------------------------------- #
    # RetinaNet
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.RETINANET.IOU_THRESHOLDS = [0.2, 0.35]  # [0.4, 0.5]
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2  # 0.02
    cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
    cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 1.0  # 0.1
    cfg.MODEL.RETINANET.NUM_CONVS = 1
