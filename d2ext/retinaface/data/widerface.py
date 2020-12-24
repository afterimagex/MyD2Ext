# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import os.path as osp

from detectron2.data.datasets import register_coco_instances

WIDERFACE_KEYPOINT_NAMES = (
    "left_eye", "right_eye",
    "nose",
    "left_mouth", "right_mouth"
)

WIDERFACE_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_mouth", "right_mouth")
)

widerface_metadata = {
    "thing_classes": ["face"],
    "keypoint_names": WIDERFACE_KEYPOINT_NAMES,
    "keypoint_flip_map": WIDERFACE_KEYPOINT_FLIP_MAP,
}

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
widerface_train_image_root = osp.join(_root, "widerface/train/images")
widerface_train_annotation_file = osp.join(_root, "widerface/train/widerface_coco.json")
register_coco_instances("widerface_train",
                        widerface_metadata,
                        widerface_train_annotation_file,
                        widerface_train_image_root)
