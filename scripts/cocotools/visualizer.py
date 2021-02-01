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


import sys
sys.path.append('..')

import cv2
import os
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from pycocotools.coco import COCO
from cocotools.utils import draw_rect


def show_info(cocoGt):
    console = Console()

    table = Table("Item", "Info", title="COCO INFO")
    for k, v in cocoGt.dataset['info'].items():
        table.add_row(str(k), str(v))
    console.print(table)

    table = Table("Name", "Id", "SuperCategory", title="COCO CATS")
    for c in cocoGt.loadCats(cocoGt.getCatIds()):
        table.add_row(str(c['name']), str(c['id']), str(c['supercategory']))
    console.print(table)


def draw_bbox(img, annoGt, cats):
    cvtBBox = lambda x: [int(x[0]), int(x[1]), int(abs(x[0] + x[2])), int(abs(x[1] + x[3]))]

    drawed = img.copy()
    catDict = {c['id']: c['name'] for c in cats}
    for anno in annoGt:
        catId = anno['category_id']
        if catId in catDict:
            name = catDict[catId]
            bbox = cvtBBox(anno['bbox'])
            cv2.rectangle(drawed, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            drawed = draw_rect(drawed, [bbox[0], bbox[1] - 30, bbox[2], bbox[1]], bgr=[255, 255, 255])
            cv2.putText(drawed, name, (bbox[0], bbox[1] - 10), 1, 1, (0, 0, 255))

    return drawed


def show_annos(image, cocoGt, annoGt, saveIm):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    cocoGt.showAnns(annoGt)
    plt.savefig(saveIm)


def main(imgRoot, cocoGt, saveRoot):
    os.makedirs(saveRoot)
    imgIds = cocoGt.getImgIds()
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    with Progress() as progress:
        task = progress.add_task("Draw COCO Gt.", total=len(imgIds))
        for idx, imgId in enumerate(imgIds):
            annoGtIds = cocoGt.getAnnIds(imgIds=[imgId])
            annoGt = cocoGt.loadAnns(annoGtIds)
            imgInfo = cocoGt.loadImgs(ids=[imgId])[0]

            image = cv2.imread(os.path.join(imgRoot, imgInfo['file_name']))
            drawed = draw_bbox(image, annoGt, cats)
            cv2.imwrite(os.path.join(saveRoot, imgInfo['file_name']), drawed)
            # show_annos(image, cocoGt, annoGt, os.path.join(saveRoot, imgInfo['file_name']))
            progress.update(task, advance=1)


if __name__ == '__main__':
    # imgRoot = '/10t/XPC/data/CCPD_COCO/image'
    # cocoGt = COCO('/10t/XPC/data/CCPD_COCO/trainval.json')
    # saveRoot = '/10t/XPC/data/CCPD_COCO/vis'

    # imgRoot = '/10t/liubing/CenterNet-master_ip_win/data/images/val'
    # cocoGt = COCO('/10t/XPC/data/bigPlateWithWindow/val_retina.json')
    # saveRoot = '/10t/XPC/data/CCPD_COCO/vis'

    # imgRoot = '/10t/XPC/data/CCPD_COCO/rng_dlpr_paste/image'
    # cocoGt = COCO('/10t/XPC/data/CCPD_COCO/rng_dlpr_paste/trainval.json')
    # saveRoot = '/10t/XPC/data/CCPD_COCO/rng_dlpr_paste/vis'

    imgRoot = '/10t/XPC/data/bigPlate/train_1th'
    cocoGt = COCO('/10t/XPC/data/bigPlate/train_1th_2cls.json')
    saveRoot = '/10t/XPC/data/bigPlate/train_1th_2cls.vis'

    show_info(cocoGt)
    main(imgRoot, cocoGt, saveRoot)