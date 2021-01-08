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
import json
import xml.etree.ElementTree as ET
import numpy as np

from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# CATEGORIES = {
#     'iPlateRect': 1,
#     'big_iPlateRect': 2,
#     'Dangerous': 3,
#     'error_label': 4,
# }

# CATEGORIES = {'luggage_rack': 1, 'skylight': 2, 'mirrors': 3, 'iPlateRect': 4, 'window': 5}
# CATEGORIES = {'民用工程船': 1, '捕鱼船': 2, '民用工业船': 3, '木船': 4}

RESIZE = False
RESIZE_W = 384
RESIZE_H = 384


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


@lru_cache(maxsize=None)
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
    idx, img_path, xml_path = data
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    images = {
        'file_name': os.path.split(img_path),
        'id': idx,
        'height': h,
        'width': w,
    }
    annotations = [
        {
            'id': 0,
            'category_id': 0,
            'ignore': 0,
            'segmentation': [],
            'area': h * w,
            'iscrowd': 0,
            'image_id': idx,
            'name': obj['name'],
            'bbox': (lambda x: [x[0], x[1], abs(x[2] - x[0]), abs(x[3] - x[1])])(obj['bbox']),
        }
        for obj in parse_rec(xml_path)
    ]
    return images, annotations


def letter_box(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    shape = img.shape[:2]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=0 * np.random.randint(0, 4))  # resized, no border(下采样)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def _main_proc(keys, imgs_maps, xmls_maps, save_dir, jsfile, categories: dict):
    json_dict = defaultdict(list)
    for cate, cid in categories.items():
        json_dict['categories'].append({
            'supercategory': None,
            'id': cid,
            'name': cate,
        })

    image_id, boxes_id = 0, 0
    for key in keys:
        if key not in imgs_maps:
            continue

        image = cv2.imread(imgs_maps[key])
        h, w = image.shape[:2]
        if RESIZE:
            image, ratio, dw, dh = letter_box(image, RESIZE_H, RESIZE_W, (0, 0, 0))
            h, w = RESIZE_H, RESIZE_W

        json_dict['images'].append({
            'file_name': os.path.split(imgs_maps[key])[-1],
            'id': image_id,
            'height': h,
            'width': w,
        })

        for obj in parse_rec(xmls_maps[key]):
            name = obj['name']
            bbox = obj['bbox']

            if name not in categories:
                print(f'WARN: {name} not in CATEGORIES')
                continue

            # if name not in ['big_iPlateRect']:
            #     continue

            if RESIZE:
                bbox[0] = ratio * bbox[0] + dw
                bbox[1] = ratio * bbox[1] + dh
                bbox[2] = ratio * bbox[2] + dw
                bbox[3] = ratio * bbox[3] + dh

            json_dict['annotations'].append({
                'id': boxes_id,
                'category_id': categories[name],
                'ignore': 0,
                'segmentation': [[bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]],
                'area': abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3]),
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': (lambda x: [x[0], x[1], abs(x[2] - x[0]), abs(x[3] - x[1])])(bbox),
            })
            boxes_id += 1

        image_id += 1
        cv2.imwrite(os.path.join(save_dir, os.path.split(imgs_maps[key])[-1]), image)
        print('[{}/{}]'.format(image_id, len(keys)))

    with open(jsfile, 'w') as fj:
        json.dump(json_dict, fj)


def main_proc(img_dir, xml_dir, dest_dir, dest_jsfile, categories):
    imgs_maps = walk_dirs(img_dir, 'jpg')
    xmls_maps = walk_dirs(xml_dir, 'xml')
    keys = list(xmls_maps.keys())
    np.random.shuffle(keys)
    os.makedirs(dest_dir, exist_ok=True)
    _main_proc(keys, imgs_maps, xmls_maps, dest_dir, dest_jsfile, categories)


# def main_proc_v2(img_dir, xml_dir, train_dir, train_jsfile, valid_dir, valid_jsfile, categories):
#     imgs_maps = walk_dirs(img_dir, 'jpg')
#     xmls_maps = walk_dirs(xml_dir, 'xml')
#     keys = list(xmls_maps.keys())
#     num_train = int(len(keys) * 0.9)
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(valid_dir, exist_ok=True)
#     np.random.shuffle(keys)
#     _main_proc(keys[:num_train], imgs_maps, xmls_maps, train_dir, train_jsfile, categories)
#     _main_proc(keys[num_train:], imgs_maps, xmls_maps, valid_dir, valid_jsfile, categories)


if __name__ == '__main__':
    # get_categorys(
    #     img_dir='/10t/XPC/data/da_haopai',
    #     xml_dir='/10t/XPC/data/da_haopai',
    # )
    # main_proc(
    #     '/10t/liubing/CenterNet-master-sky/old_data/images/images/',
    #     '/10t/liubing/CenterNet-master-sky/old_data/images/annotations/',
    #     '/10t/XPC/data/PlateWinCoco/valid256',
    #     '/10t/XPC/data/PlateWinCoco/valid256.json',
    # )
    # main_proc_v2(
    #     '/10t/XPC/data/da_haopai',
    #     '/10t/XPC/data/da_haopai',
    #     '/10t/XPC/data/bigPlate/train',
    #     '/10t/XPC/data/bigPlate/train.json',
    #     '/10t/XPC/data/bigPlate/valid',
    #     '/10t/XPC/data/bigPlate/valid.json',
    # )

    # get_categorys(
    #     img_dir='/10t/XPC/data/second_biaozhu/second_biaozhu',
    #     xml_dir='/10t/XPC/data/second_biaozhu/second_biaozhu',
    # )

    plate_categories = {
        'iPlateRect': 1,
        'big_iPlateRect': 2,
        'Dangerous': 3,
        'error_label': 4,
    }

    main_proc(
        '/10t/XPC/data/da_haopai',
        '/10t/XPC/data/da_haopai',
        '/10t/XPC/data/bigPlate/train_1th',
        '/10t/XPC/data/bigPlate/train_1th.json',
        categories=plate_categories,
    )

    main_proc(
        '/10t/XPC/data/second_biaozhu/second_biaozhu',
        '/10t/XPC/data/second_biaozhu/second_biaozhu',
        '/10t/XPC/data/bigPlate/train_2th',
        '/10t/XPC/data/bigPlate/train_2th.json',
        categories=plate_categories,
    )

    # main_proc(
    #     '/10t/XPC/data/da_haopai',
    #     '/10t/XPC/data/da_haopai',
    #     '/10t/XPC/data/bigPlate/train_1th_384',
    #     '/10t/XPC/data/bigPlate/train_1th_384.json',
    # )
    #
    # main_proc(
    #     '/10t/XPC/data/second_biaozhu/second_biaozhu',
    #     '/10t/XPC/data/second_biaozhu/second_biaozhu',
    #     '/10t/XPC/data/bigPlate/train_2th_384',
    #     '/10t/XPC/data/bigPlate/train_2th_384.json',
    # )

    # get_categorys(
    #     img_dir='/10t/XPC/data/ship/data1',
    #     xml_dir='/10t/XPC/data/ship/data1',
    # )
    #
    # main_proc(
    #     '/10t/XPC/data/ship/data1',
    #     '/10t/XPC/data/ship/data1',
    #     '/10t/XPC/data/ship/coco',
    #     '/10t/XPC/data/ship/coco.json',
    # )
