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


import argparse
import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

logger = Console()

from pycococreatortools import create_image_info, create_annotation_info_polygon

INFO = {
    "description": "CCPD Dataset in COCO Format",
    "url": "",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Tristan, Onlyyou intern",
    "date_created": datetime.datetime.utcnow().isoformat(' ')  # 显示此刻时间，格式：'2019-04-30 02:17:49.040415'
}

LICENSES = [
    {
        "id": 1,
        "name": "ALL RIGHTS RESERVED",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'plate',
        'supercategory': 'shape',
    },
]


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None, type=str, required=True,
                        help="The input data dir. Should contain all the images")
    parser.add_argument("-o", "--output", default=None, type=str, required=True,
                        help="The output json file.")
    return parser.parse_args()


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


def parse_ccpd_filename(inputs):
    try:
        image = Image.open(inputs['ImgPath'].absolute())
        annos = inputs['ImgPath'].name.split('.')[0].split('-')
        lt, rb = [[float(eel) for eel in el.split('&')] for el in annos[2].split('_')]
        nh, nw = abs(rb[1] - lt[1]), abs(rb[0] - lt[0])
        bbox = [lt[0], lt[1], nw, nh]
        poly = [[float(eel) for eel in el.split('&')] for el in annos[3].split('_')]
        poly = sort_polygon(np.array(poly))
        kpts = np.array([[p[0], p[1], 2] for p in poly])

        if not inputs['Extend'] is None:
            bbox, poly = inputs['Extend'](inputs['ImgPath'], inputs['SavePath'], bbox, poly)

        area = compute_polygon_area(poly)

        image_info = create_image_info(
            image_id=0,
            file_name=inputs['ImgPath'].name,
            image_size=image.size,
        )
        annotation_info = create_annotation_info_polygon(
            annotation_id=0,
            image_id=0,
            area=area,
            category_id=1,
            image_size=image.size,
            bounding_box=bbox,
            segmentation=poly.tolist(),
            num_keypoints=4,
            keypoints=kpts.flatten().tolist(),
        )
        return image_info, annotation_info

    except Exception:
        logger.print_exception()
        return None, None


def main(args):
    Path(args.output).mkdir(exist_ok=True)
    Path(args.output).joinpath('image').mkdir(exist_ok=True)
    im_save_dir = Path(args.output).joinpath('image')

    im_files = [
        {'ImgPath': pt, 'SavePath': im_save_dir, 'Extend': None}
        for pt in Path(args.input).rglob('*.jpg')
        if len(pt.name.split('-')) > 3
    ]

    json_dict = {
        'info': INFO,
        'licenses': LICENSES,
        'categories': CATEGORIES,
        'images': [],
        'annotations': []
    }

    annotation_id = 0
    executor = ThreadPoolExecutor(max_workers=10)

    with Progress() as progress:
        task = progress.add_task("{Convert CCPD to COCO Format.}", total=len(im_files))
        # for i, files in enumerate(im_files):
        #     image_info, annotation_info = parse_ccpd_filename(files)
        for i, (image_info, annotation_info) in enumerate(executor.map(parse_ccpd_filename, im_files)):
            if image_info is None or annotation_info is None:
                continue

            image_info.update({"id": i})
            annotation_info.update({"id": annotation_id, 'image_id': i})
            annotation_id += 1

            json_dict['images'].append(image_info)
            json_dict['annotations'].append(annotation_info)

            progress.update(task, advance=1)
            if i % 1000 == 0:
                table = Table("ImageId", "AnnotationId", "Segmentation", "Area",
                              title=":smiley: Process on [{}/{}]".format(i, len(im_files)))
                table.add_row(
                    str(image_info['id']),
                    str(annotation_info['id']),
                    ','.join(list(map(lambda x: str(x), annotation_info['segmentation']))),
                    str(annotation_info['area']),
                )
                progress.console.print(table)

        with Path(args.output).joinpath('trainval.json').open('w') as fj:
            json.dump(json_dict, fj)


if __name__ == "__main__":
    args = default_parser()
    main(args)
