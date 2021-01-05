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
    "keypoint_names": ["lt", "rt", "rb", "lb"],
    "keypoint_flip_map": [["lt", "rt"], ["lb", "rb"]],
}

register_coco_instances(
    name="ccpd_base_train",
    metadata=ccpd_metadata,
    json_file="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/base/trainval.oks.json",
    image_root="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/base/image",
)

register_coco_instances(
    name="ccpd_danger_train",
    metadata=ccpd_metadata,
    json_file="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_danger_past/trainval.oks.json",
    image_root="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_danger_past/image",
)

register_coco_instances(
    name="ccpd_dlpr_train",
    metadata=ccpd_metadata,
    json_file="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_dlpr_paste/trainval.oks.json",
    image_root="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_dlpr_paste/image",
)

register_coco_instances(
    name="ccpd_slpr1_train",
    metadata=ccpd_metadata,
    json_file="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_slpr_paste1/trainval.oks.json",
    image_root="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_slpr_paste1/image",
)

register_coco_instances(
    name="ccpd_slpr_train",
    metadata=ccpd_metadata,
    json_file="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_slpr_paste/trainval.oks.json",
    image_root="/media/ps/A1/XPC/data/CCPD/CCPD_COCO/rng_slpr_paste/image",
)
