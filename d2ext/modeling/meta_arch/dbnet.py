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


import torch
from torch import nn

from d2ext.modeling.centernet_gt import CenterNetGT
from d2ext.utils.visualizer import TrainingVisualizer
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling.backbone.build import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess

__all__ = ["DBNet"]


@META_ARCH_REGISTRY.register()
class DBNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            upsample,
            head,
            num_classes,
            gt_generator,
            pixel_mean,
            pixel_std,
            max_detections_per_image=100,
            classification_loss_type,
            regression_loss_type,
            head_in_features,
            test_score_thresh,
            hm_norm,
            hm_weight,
            wh_weight,
            reg_weight,
            kd_enable=False,
            kd_loss,
            kd_without_label,
            vis_period=0,
            input_format="BGR",
            visualizer,
    ):
        super().__init__()

        self.backbone = backbone
        self.upsample = upsample
        self.head = head
        self.head_in_features = head_in_features

        self.num_classes = num_classes
        self.gt_generator = gt_generator

        # Inference parameters:
        self.max_detections_per_image = max_detections_per_image
        self.test_score_thresh = test_score_thresh

        # Loss parameters:
        self.classification_loss_type = classification_loss_type
        self.regression_loss_type = regression_loss_type

        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format
        self.visualizer = visualizer

        if kd_enable:
            self.kd_loss = kd_loss
            self.kd_without_label = kd_without_label

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        self.hm_norm = hm_norm
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        feature_shapes = backbone.output_shape()[cfg.MODEL.CENTERNET.IN_FEATURES]
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused"
        )
        return {
            "backbone": backbone,
            "upsample": CenternetDeconv(cfg, feature_shapes),
            "head": CenternetHead(cfg),
            "head_in_features": cfg.MODEL.CENTERNET.IN_FEATURES,
            "num_classes": cfg.MODEL.CENTERNET.NUM_CLASSES,
            "test_score_thresh": cfg.MODEL.CENTERNET.SCORE_THRESH_TEST,
            "gt_generator": CenterNetGT(
                cfg.MODEL.CENTERNET.NUM_CLASSES,
                cfg.MODEL.CENTERNET.DOWN_SCALE,
                cfg.MODEL.CENTERNET.MIN_OVERLAP,
                cfg.MODEL.CENTERNET.TENSOR_DIM,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "classification_loss_type": cfg.MODEL.CENTERNET.LOSS.IGNORE_UNLABEL,
            "regression_loss_type": cfg.MODEL.CENTERNET.LOSS.MSE,
            "hm_norm": cfg.MODEL.CENTERNET.LOSS.NORM_WH,
            "hm_weight": cfg.MODEL.CENTERNET.LOSS.HM_WEIGHT,
            'wh_weight': cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT,
            'reg_weight': cfg.MODEL.CENTERNET.LOSS.REG_WEIGHT,
            "kd_enable": cfg.MODEL.CENTERNET.KD.ENABLED,
            "kd_loss": None,
            "kd_without_label": cfg.MODEL.CENTERNET.KD.KD_WITHOUT_LABEL,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
            "visualizer": TrainingVisualizer(detector_postprocess, metadata)
        }

    @property
    def device(self):
        return self.pixel_mean.device