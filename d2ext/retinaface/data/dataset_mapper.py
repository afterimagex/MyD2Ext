# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.data.dataset_mapper import DatasetMapper as BaseDatasetMapper
from . import detection_utils as d_utils

class DatasetMapper(BaseDatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.tfm_gens = d_utils.build_transform_gen(cfg, is_train)
