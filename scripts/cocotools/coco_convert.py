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


import json

import numpy as np


def sort_polygon(poly):
    # lt, rt, rb, lb
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    p_area = np.sum(edge) / 2.

    _poly = poly.copy()
    if abs(p_area) < 1:
        raise ValueError
    if p_area > 0:
        _poly = _poly[(0, 3, 2, 1), :]  # clock wise

    anchor = np.array([np.min(poly[:, 0]), np.min(poly[:, 1])])
    line0 = np.linalg.norm(anchor - _poly[0])
    line1 = np.linalg.norm(anchor - _poly[1])
    line2 = np.linalg.norm(anchor - _poly[2])
    line3 = np.linalg.norm(anchor - _poly[3])

    argmin = np.argmin([line0, line1, line2, line3])

    lt = _poly[argmin]
    rt = _poly[(argmin + 1) % 4]
    rb = _poly[(argmin + 2) % 4]
    lb = _poly[(argmin + 3) % 4]

    return np.array([lt, rt, rb, lb])


def category_filter(json_file, save_file, keep_cats: dict):
    def __update(x, d):
        x.update(d)
        return x

    with open(json_file, 'r') as fin:
        json_dict = json.load(fin)
        new_categories = [
            __update(cat, {'id': keep_cats[cat['name']]})
            for cat in json_dict['categories']
            if cat['name'] in keep_cats.keys()
        ]
        new_annotations = [
            anno
            for anno in json_dict['annotations']
            if int(anno['category_id']) in keep_cats.values()
        ]
        json_dict['categories'] = new_categories
        json_dict['annotations'] = new_annotations

    print(json_dict['categories'])
    with open(save_file, 'w') as fout:
        json.dump(json_dict, fout)


def range_gather(json_file, save_file, start, stop):
    with open(json_file, 'r') as fin:
        json_dict = json.load(fin)
        new_images = [
            img
            for img in json_dict['images']
            if start <= int(img['id']) < stop
        ]
        new_annotations = [
            anno
            for anno in json_dict['annotations']
            if start <= int(anno['id']) < stop
        ]

        json_dict['images'] = new_images
        json_dict['annotations'] = new_annotations

    print(json_dict['categories'])
    print(len(json_dict['images']))
    print(len(json_dict['annotations']))
    with open(save_file, 'w') as fout:
        json.dump(json_dict, fout)


def category_update(json_file, save_file, keep_cats: dict):
    cats_map = {}
    new_categories = []
    new_annotations = []

    with open(json_file, 'r') as fin:
        json_dict = json.load(fin)

        for cat in json_dict['categories']:
            if cat['name'] in keep_cats.keys():
                catId = keep_cats[cat['name']]
                cats_map[cat['id']] = catId
                cat.update({'id': catId})
                new_categories.append(cat)

        for anno in json_dict['annotations']:
            catId = anno['category_id']
            if catId in cats_map.keys():
                anno.update({'category_id': cats_map[catId]})
                new_annotations.append(anno)

        json_dict['categories'] = new_categories
        json_dict['annotations'] = new_annotations

    print(json_dict['categories'])
    print(len(json_dict['annotations']))
    with open(save_file, 'w') as fout:
        json.dump(json_dict, fout)


def category_filter_oks(json_file, save_file):
    def __update(anno):
        polys = sort_polygon(np.array(anno['segmentation']))
        polys = np.array([[p[0], p[1], 2] for p in polys])
        anno['segmentation'] = None
        anno['num_keypoints'] = 4
        anno['keypoints'] = polys.flatten().tolist()
        return anno

    with open(json_file, 'r') as fin:
        json_dict = json.load(fin)
        new_annotations = [
            __update(anno)
            for anno in json_dict['annotations']
        ]
        json_dict['annotations'] = new_annotations

    print(json_dict['categories'])
    with open(save_file, 'w') as fout:
        json.dump(json_dict, fout)


if __name__ == '__main__':
    # category_filter(
    #     '/10t/XPC/data/bigPlate/train_1th_384.json',
    #     '/10t/XPC/data/bigPlate/train_1th_384_2cls.json',
    #     {'iPlateRect': 1, 'big_iPlateRect': 2},
    # )
    # category_filter(
    #     '/10t/XPC/data/bigPlate/train_2th_384.json',
    #     '/10t/XPC/data/bigPlate/train_2th_384_2cls.json',
    #     {'iPlateRect': 1, 'big_iPlateRect': 2},
    # )
    # category_filter(
    #     '/10t/XPC/data/bigPlate/train_1th.json',
    #     '/10t/XPC/data/bigPlate/train_1th_2cls.json',
    #     {'iPlateRect': 1, 'big_iPlateRect': 2},
    # )
    # category_filter(
    #     '/10t/XPC/data/bigPlate/train_2th.json',
    #     '/10t/XPC/data/bigPlate/train_2th_2cls.json',
    #     {'iPlateRect': 1, 'big_iPlateRect': 2},
    # )
    # category_filter(
    #     '/10t/XPC/data/bigPlate/train_1th.json',
    #     '/10t/XPC/data/bigPlate/train_1th_1cls.json',
    #     {'big_iPlateRect': 1},
    # )
    # category_filter(
    #     '/10t/XPC/data/bigPlate/train_2th.json',
    #     '/10t/XPC/data/bigPlate/train_2th_1cls.json',
    #     {'big_iPlateRect': 1},
    # )
    range_gather(
        '/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.json',
        '/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.lite.json',
        0, 100
    )
    # category_update(
    #     '/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.lite.json',
    #     '/media/ps/A1/XPC/data/CCPD/ccpd_rotate_coco/trainval.lite1.json',
    #     {'plate': 1},
    # )
