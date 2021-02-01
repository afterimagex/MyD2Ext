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

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from pascal_voc_writer import Writer

CROP_SIZE = (320, 320)
STRIDE = 160


def walk_dirs(paths, suffix):
    dir_map = {}

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        for (root, dirs, files) in os.walk(path):
            for item in files:
                if item.endswith(suffix):
                    d = os.path.abspath(os.path.join(root, item))
                    dir_map[item.split('.')[0]] = d
    return dir_map


def iou(box, boxes, thresh=0.5):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (area + areas - inter)
    keep = np.where(ovr >= thresh)[0]
    return keep


def ioa(box, boxes, thresh=0.5):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / areas
    keep = np.where(ovr >= thresh)[0]
    return keep


@lru_cache(maxsize=500)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    return [
        {
            'name': obj.find('name').text.strip(),
            'pose': obj.find('pose').text.strip(),
            'truncated': int(obj.find("truncated").text),
            'difficult': int(obj.find("difficult").text),
            'bbox': [float(obj.find('bndbox').find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        }
        for obj in ET.parse(filename).findall('object')
    ]


def make_grids(img_size, sub_size, stride):
    img_w, img_h = img_size
    sub_w, sub_h = sub_size
    xs = (np.arange(img_w) + 0.5) * stride
    ys = (np.arange(img_h) + 0.5) * stride
    xs = xs + (sub_w - stride) / 2.0
    ys = ys + (sub_h - stride) / 2.0
    xs = xs[np.where(xs < img_w)]
    ys = ys[np.where(ys < img_h)]
    xs = [sub_w / 2.0] if len(xs) == 0 else xs
    ys = [sub_h / 2.0] if len(ys) == 0 else ys
    # print(xs, ys)
    X, Y = np.meshgrid(xs, ys)
    # print(X, Y)
    # plt.plot(X, Y, color='limegreen', marker='.', linestyle='')
    # plt.grid(True)
    # plt.show()
    return np.vstack([
        X.ravel() - sub_w / 2.0,
        Y.ravel() - sub_h / 2.0,
        X.ravel() + sub_w / 2.0,
        Y.ravel() + sub_h / 2.0,
    ]).T


def process(data):
    idx, img_path, xml_path, dst_dir = data
    annos = parse_rec(xml_path)
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    grid = make_grids((w, h), CROP_SIZE, STRIDE)
    boxes = np.array([x['bbox'] for x in annos])
    for i, (gx1, gy1, gx2, gy2) in enumerate(np.int32(grid)):
        dx1, dx2 = gx1, min(gx2, w)
        dy1, dy2 = gy1, min(gy2, h)
        roi = image[dy1: dy2, dx1: dx2]
        empty = np.zeros((gy2 - gy1, gx2 - gx1, 3), dtype=np.uint8)
        empty[:roi.shape[0], :roi.shape[1]] = roi
        keep = ioa([gx1, gy1, gx2, gy2], boxes)
        if len(keep) > 0:
            imfile = os.path.join(dst_dir, 'img', '{}_{}_{}_{}_{}.jpg'.format(i, dx1, dy1, dx2, dy2))
            xmfile = os.path.join(dst_dir, 'xml', '{}_{}_{}_{}_{}.xml'.format(i, dx1, dy1, dx2, dy2))
            writer = Writer(
                path=imfile,
                width=roi.shape[1],
                height=roi.shape[0],
                database='None',
            )
            for j in keep:
                name = annos[j]['name']
                kbox = list(map(int, boxes[j]))
                writer.addObject(name, kbox[0], kbox[1], kbox[2], kbox[3])
            writer.save(xmfile)
            cv2.imwrite(imfile, roi)


def main(img_dir, xml_dir, dst_dir):
    imgs_maps = walk_dirs(img_dir, 'png')
    xmls_maps = walk_dirs(xml_dir, 'xml')
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'xml'), exist_ok=True)

    data = [
        (idx, imgs_maps[key], xmls_maps[key], dst_dir)
        for idx, key in enumerate(xmls_maps.keys())
        if key in imgs_maps
    ]

    executor = ThreadPoolExecutor(max_workers=20)
    executor.map(process, data)


def test_ioa():
    keep = ioa(
        box=[100, 100, 500, 500],
        boxes=np.array([
            [0, 0, 100, 100],
            [100, 100, 400, 400],
            [100, 100, 500, 500],
            [100, 100, 800, 900],
        ])
    )
    print(keep)


if __name__ == '__main__':
    # test_ioa()
    main(
        img_dir='/10t/XPC/dongche/image',
        xml_dir='/10t/XPC/dongche/xml',
        dst_dir='/10t/XPC/dongche/crop320',
    )
