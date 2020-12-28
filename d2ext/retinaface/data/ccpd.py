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


from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    name="ccpd_rotate_train",
    metadata={
        "thing_classes": ["plate"],
        "keypoint_names": ["lt", "rt", "rb", "lb"],
        "keypoint_flip_map": [["lt", "rt"], ["lb", "rb"]],
    },
    json_file="/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.json",
    image_root="/media/ps/A1/XPC/data/CCPD/ccpd_rotate",
)
