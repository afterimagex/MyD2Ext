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


import struct
from itertools import product as product
from math import ceil

import numpy as np
import torch

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.structures.boxes import Boxes


class OriginPriorBox(object):
    def __init__(self, image_size=(800, 800)):
        super(OriginPriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size  # / self.image_size[1]
                    s_ky = min_size  # / self.image_size[0]
                    dense_cx = [x * self.steps[k] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def load_priors(priors_filename):
    with open(priors_filename, "rb") as f:
        dims = struct.unpack("=i", f.read(4))[0]
        shape = []
        for i in range(dims):
            shape.append(struct.unpack("=i", f.read(4))[0])
        count = np.prod(shape)
        data = []
        for i in range(count):
            data.append(struct.unpack("=f", f.read(4))[0])

        return np.asarray(data, dtype=np.float32).reshape(shape)


if __name__ == '__main__':
    priorbox0 = OriginPriorBox().forward().numpy()
    print(priorbox0, priorbox0.shape)

    # priorbox1 = load_priors('/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/output/model_0335999.anc')
    # print(priorbox1, priorbox1.shape)
    fmap = [
        torch.randn(1, 3, 100, 100),
        torch.randn(1, 3, 50, 50),
        torch.randn(1, 3, 25, 25),
    ]
    dag = DefaultAnchorGenerator(
        sizes=[[16, 32], [64, 128], [256, 512]],
        aspect_ratios=[[1.0]],
        strides=[8, 16, 32],
        offset=0.5,
    )
    anc = dag(fmap)
    anc = Boxes.cat(anc).tensor.detach().cpu().numpy()
    print(anc, anc.shape)
