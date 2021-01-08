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

ccpd_metadata = {
    "thing_classes": ["plate"],
}

register_coco_instances(
    name="ccpd_rotate_train",
    metadata=ccpd_metadata,
    json_file="/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.lite.json",
    image_root="/media/ps/A1/XPC/data/CCPD/ccpd_rotate",
)
