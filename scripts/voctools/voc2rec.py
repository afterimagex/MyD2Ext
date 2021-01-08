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
import os
import numpy as np
import xml.etree.ElementTree as ET

from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


# from paddle_rec import RecNet
#
# NET = RecNet(
#     '/10t/XPC/models/rec/',
#     '/10t/XPC/models/rec/charsets.txt',
# )


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


def _get_categorys(data):
    idx, img_path, xml_path = data
    names = [
        obj.find('name').text.strip()
        for obj in ET.parse(xml_path).findall('object')
    ]
    return names


def get_categorys(img_dir, xml_dir):
    imgs_maps = walk_dirs(img_dir, 'jpg')
    xmls_maps = walk_dirs(xml_dir, 'xml')
    data = [
        (idx, imgs_maps[key], xmls_maps[key])
        for idx, key in enumerate(xmls_maps.keys())
        if key in imgs_maps
    ]
    categorys = defaultdict(int)
    executor = ThreadPoolExecutor(max_workers=20)
    for cate in executor.map(_get_categorys, data):
        for c in cate:
            categorys[c] += 1
    print(categorys)


@lru_cache(maxsize=200)
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


def process(data):
    idx, img_path, xml_path, save_dir = data
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    for j, obj in enumerate(parse_rec(xml_path)):
        if obj['name'] not in ['big_iPlateRect']:
            continue
        xmin, ymin, xmax, ymax = obj['bbox']
        xmin = int(np.clip(xmin, 0, w - 1))
        ymin = int(np.clip(ymin, 0, h - 1))
        xmax = int(np.clip(xmax, 0, w - 1))
        ymax = int(np.clip(ymax, 0, h - 1))
        roi = image[ymin:ymax, xmin:xmax]
        # text = NET.predict(roi)
        cv2.imwrite(os.path.join(save_dir, f'{idx}_{j}_Null.jpg'), roi)

    print(idx)


def main(img_dir, xml_dir, save_dir):
    imgs_maps = walk_dirs(img_dir, 'jpg')
    xmls_maps = walk_dirs(xml_dir, 'xml')

    data = [
        (i, imgs_maps[key], xmls_maps[key], save_dir)
        for i, key in enumerate(xmls_maps.keys())
        if key in imgs_maps
    ]

    executor = ThreadPoolExecutor(max_workers=10)
    executor.map(process, data)


if __name__ == '__main__':
    # get_categorys(
    #     img_dir='/10t/XPC/data/da_haopai',
    #     xml_dir='/10t/XPC/data/da_haopai',
    # )
    # main(
    #     '/10t/XPC/data/da_haopai',
    #     '/10t/XPC/data/da_haopai',
    #     '/10t/XPC/data/bigPlate/roi'
    # )

    main(
        '/10t/XPC/data/second_biaozhu/second_biaozhu',
        '/10t/XPC/data/second_biaozhu/second_biaozhu',
        '/10t/XPC/data/bigPlate/roi_2th'
    )
