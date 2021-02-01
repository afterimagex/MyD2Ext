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
    },
    json_file="/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.lite.json",
    image_root="/media/ps/A1/XPC/data/CCPD/ccpd_rotate",
)

register_coco_instances(
    name="truck_train",
    metadata={
        "thing_classes": ["truck"],
    },
    json_file="/media/ps/A1/XPC/data/truck/annotations/train_one_2020.json",
    image_root="/media/ps/A1/XPC/data/truck/images",
)

register_coco_instances(
    name="truck_valid",
    metadata={
        "thing_classes": ["truck"],
    },
    json_file="/media/ps/A1/XPC/data/truck/annotations/val_one_2020.json",
    image_root="/media/ps/A1/XPC/data/truck/images",
)

register_coco_instances(
    name="fsbj_1th",
    metadata={
        "thing_classes": ["fsbj"],
    },
    json_file="/media/ps/A1/XPC/data/fsbj/train_fsbj_320_1th.json",
    image_root="/media/ps/A1/XPC/data/fsbj/train_fsbj_320_1th",
)
