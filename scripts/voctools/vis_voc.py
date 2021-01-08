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

import os
import cv2
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor


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


def main_draw(data):
    img_path, xml_path, save_path = data
    image = cv2.imread(img_path)
    for obj in ET.parse(xml_path).findall('object'):
        name = obj.find('name').text.strip().lower()
        x1, y1, x2, y2 = [int(float(obj.find('bndbox').find(x).text)) for x in ["xmin", "ymin", "xmax", "ymax"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.rectangle(image, (x1, y1 - 20), (x2, y1), (255, 255, 0), -1)
        cv2.putText(image, name, (x1, y1), 1, 1, (0, 0, 0))
    cv2.imwrite(save_path, image)
    print(save_path)


if __name__ == '__main__':
    save_dir = '/10t/liubing/CenterNet-master-sky/old_data/images/vis'
    os.makedirs(save_dir, exist_ok=True)

    xmls_maps = walk_dirs('/10t/liubing/CenterNet-master-sky/old_data/images/annotations/', 'xml')
    imgs_maps = walk_dirs('/10t/liubing/CenterNet-master-sky/old_data/images/images/', 'jpg')

    ixs = []
    for key in xmls_maps.keys():
        if key not in imgs_maps:
            continue
        save_path = os.path.join(save_dir, f'{key}.jpg')
        ixs.append((imgs_maps[key], xmls_maps[key], save_path))

    # executor = ThreadPoolExecutor(max_workers=1)
    # executor.map(main_draw, ixs)
    for d in ixs:
        main_draw(d)
