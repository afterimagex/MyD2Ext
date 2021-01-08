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

import numpy as np


def draw_rect(img, bbox, bgr, alpha=0.5):
    x1, y1, x2, y2 = bbox
    img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * alpha + np.array(bgr) * (1 - alpha)
    return img
