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
import numpy as np


# 根据车牌的四个角点绘制精确的segmentation map
def make_seg_mask(map, segmentions, color=(0, 255, 0)):
    c = np.array([[segmentions]], dtype=np.int32)
    cv2.fillPoly(map, c, color)


def random_color(class_id):
    '''
    预定义12种颜色，基本涵盖kjdz所有label类型
    颜色对照网址：https://tool.oschina.net/commons?type=3
    '''
    colorArr = [
        (255, 0, 0),  # 红色
        (255, 255, 0),  # 黄色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (160, 32, 240),  # 紫色
        (165, 42, 42),  # 棕色
        (238, 201, 0),  # gold
        (255, 110, 180),  # HotPink1
        (139, 0, 0),  # DarkRed
        (0, 139, 139),  # DarkCyan
        (139, 0, 139),  # DarkMagenta
        (0, 0, 139)  # dark blue
    ]
    if class_id < 11:
        return colorArr[class_id]
    else:  # 如有特殊情况，类别数超过12，则随机返回一个颜色
        rm_col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        return rm_col


# 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
def compute_polygon_area(points):
    point_num = len(points)
    if point_num < 3:
        return 0.0
    s = points[0][1] * (points[point_num - 1][0] - points[1][0])
    # for i in range(point_num): # (int i = 1 i < point_num ++i):
    for i in range(1, point_num):
        s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
    return abs(s / 2.0)
