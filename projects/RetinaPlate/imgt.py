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

import cv2


def CenterPadResize(image, bchw_shape, border_value):
    src_h, src_w = image.shape[:2]
    _, _, dst_h, dst_w = bchw_shape
    ratio = min(float(dst_h) / src_h, float(dst_w) / src_w)
    new_size = (round(src_w * ratio), round(src_h * ratio))
    dw = (dst_w - new_size[0]) / 2
    dh = (dst_h - new_size[1]) / 2
    t, b = round(dh - 0.1), round(dh + 0.1)
    l, r = round(dw - 0.1), round(dw + 0.1)
    image = cv2.resize(image, new_size, interpolation=0)
    image = cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=border_value)
    return image


image = cv2.imread(
    '/media/ps/A1/XPC/data/CCPD/ccpd_rotate/1316-21_16-167&353_530&656-530&519_183&656_167&490_514&353-0_0_6_27_27_30_22-47-40.jpg')

image = CenterPadResize(image, (1, 3, 640, 640), 0)

cv2.rectangle(image, (200, 190), (385, 333), (255, 0, 255), 1)
cv2.rectangle(image, (201, 194), (381, 333), (255, 0, 255), 1)
cv2.rectangle(image, (200, 188), (386, 337), (255, 0, 255), 1)

cv2.imwrite('123.jpg', image)
