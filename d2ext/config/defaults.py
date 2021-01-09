# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Copyright (C) 2020-Present, Pvening, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------


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


def add_centernet_config(cfg):
    """
    Add config for tridentnet.
    """

    # centernet config
    cfg.MODEL.CENTERNET = CN()
    cfg.MODEL.CENTERNET.IN_FEATURES = "res5"

    cfg.MODEL.CENTERNET.NUM_CLASSES = 80
    cfg.MODEL.CENTERNET.BIAS_VALUE = -2.19
    cfg.MODEL.CENTERNET.DOWN_SCALE = 4
    cfg.MODEL.CENTERNET.MIN_OVERLAP = 0.7
    cfg.MODEL.CENTERNET.TENSOR_DIM = 128
    cfg.MODEL.CENTERNET.BOX_MINSIZE = 1e-5
    cfg.MODEL.CENTERNET.SCORE_THRESH_TEST = 0.05

    cfg.MODEL.CENTERNET.DECONV = CN()
    cfg.MODEL.CENTERNET.DECONV.CHANNEL = [256, 128, 64]
    cfg.MODEL.CENTERNET.DECONV.KERNEL = [4, 4, 4]
    cfg.MODEL.CENTERNET.DECONV.DEFORM = False
    cfg.MODEL.CENTERNET.DECONV.DEFORM_MODULATED = False

    # cfg.MODEL.CENTERNET.RESIZE_TYPE = "ResizeShortestEdge"
    # cfg.MODEL.CENTERNET.TRAIN_PIPELINES = [
    #     # ("CenterAffine", dict(boarder=128, output_size=(512, 512), random_aug=True)),
    #     ("RandomFlip", dict()),
    #     ("RandomBrightness", dict(intensity_min=0.6, intensity_max=1.4)),
    #     ("RandomContrast", dict(intensity_min=0.6, intensity_max=1.4)),
    #     ("RandomSaturation", dict(intensity_min=0.6, intensity_max=1.4)),
    #     ("RandomLighting", dict(scale=0.1)),
    # ]
    cfg.MODEL.CENTERNET.TEST_PIPELINES = []
    cfg.MODEL.CENTERNET.LOSS = CN()
    cfg.MODEL.CENTERNET.LOSS.HM_WEIGHT = 1
    cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT = 0.1
    cfg.MODEL.CENTERNET.LOSS.REG_WEIGHT = 1
    cfg.MODEL.CENTERNET.LOSS.NORM_WH = False
    cfg.MODEL.CENTERNET.LOSS.SKIP_LOSS = False
    cfg.MODEL.CENTERNET.LOSS.SKIP_WEIGHT = 1.0
    cfg.MODEL.CENTERNET.LOSS.MSE = False
    cfg.MODEL.CENTERNET.LOSS.IGNORE_UNLABEL = False

    cfg.MODEL.CENTERNET.LOSS.COMMUNISM = CN()
    cfg.MODEL.CENTERNET.LOSS.COMMUNISM.ENABLE = False
    cfg.MODEL.CENTERNET.LOSS.COMMUNISM.CLS_LOSS = 1.5
    cfg.MODEL.CENTERNET.LOSS.COMMUNISM.WH_LOSS = 0.3
    cfg.MODEL.CENTERNET.LOSS.COMMUNISM.OFF_LOSS = 0.1
    cfg.MODEL.CENTERNET.IMGAUG_PROB = 2.0

    # optim and min_lr(for cosine schedule)
    cfg.SOLVER.MIN_LR = 1e-8
    cfg.SOLVER.OPTIM_NAME = "SGD"
    cfg.SOLVER.COSINE_DECAY_ITER = 0.7

    # Knowledge Distill
    cfg.MODEL.CENTERNET.KD = CN()
    cfg.MODEL.CENTERNET.KD.ENABLED = False
    cfg.MODEL.CENTERNET.KD.TEACHER_CFG = ["None", ]
    cfg.MODEL.CENTERNET.KD.TEACHER_WEIGTHS = ["None", ]
    cfg.MODEL.CENTERNET.KD.KD_WEIGHT = [10.0, ]
    cfg.MODEL.CENTERNET.KD.KD_CLS_WEIGHT = [1.0, ]
    cfg.MODEL.CENTERNET.KD.KD_WH_WEIGHT = [1.0, ]
    cfg.MODEL.CENTERNET.KD.KD_REG_WEIGHT = [0.1, ]
    cfg.MODEL.CENTERNET.KD.KD_CLS_INDEX = [1, ]
    cfg.MODEL.CENTERNET.KD.NORM_WH = [False, ]
    cfg.MODEL.CENTERNET.KD.KD_WITHOUT_LABEL = False

    # input config
    cfg.INPUT.FORMAT = "RGB"
