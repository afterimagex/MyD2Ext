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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# {'luggage_rack': 1, 'skylight': 2, 'mirrors': 3, 'iPlateRect': 4, 'window': 5}

plt.rcParams['savefig.dpi'] = 500


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def _draw_rect(img, bbox, bgr, alpha=0.5):
    x1, y1, x2, y2 = bbox
    img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * alpha + np.array(bgr) * (1 - alpha)
    return img


def drawV2(img, annoGt, annoDt, cats, badStr):
    cvtBBox = lambda x: [int(x[0]), int(x[1]), int(abs(x[0] + x[2])), int(abs(x[1] + x[3]))]

    # left = img.copy() * 0.6
    right = img.copy() * 0.6
    catDict = {c['id']: c['name'] for c in cats}
    # for anno in annoGt:
    #     catId = anno['category_id']
    #     if catId in catDict:
    #         name = catDict[catId]
    #         bbox = cvtBBox(anno['bbox'])
    #         cv2.rectangle(left, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
    #         left = _draw_rect(left, [bbox[0], bbox[1] - 30, bbox[2], bbox[1]], bgr=[255, 255, 255])
    #         cv2.putText(left, name, (bbox[0], bbox[1] - 10), 1, 1, (0, 0, 255))

    for anno in annoDt:
        catId = anno['category_id']
        if catId in catDict:
            bbox = cvtBBox(anno['bbox'])
            cv2.rectangle(right, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            right = _draw_rect(right, [bbox[0], bbox[1] - 30, bbox[2], bbox[1]], bgr=[255, 255, 255])
            cv2.putText(right, f'{anno["score"]:.2f}_{catDict[catId]}', (bbox[0], bbox[1] - 10), 1, 1, (0, 0, 255))

    # dst = np.hstack([img, left, right])
    cv2.putText(right, badStr, (35, 35), 1, 1, (0, 0, 255))
    return right


def evaluate_predictions_on_coco(cocoEval, cats):
    '''
    绘制PR曲线
    第一维T：IoU的10个阈值，从0.5到0.95间隔0.05。
    第二维R：101个recall 阈值，从0到101
    第三维K：类别，如果是想展示第一类的结果就设为0
    第四维A：area 目标的大小范围 （all，small, medium, large）
    第五维M：maxDets 单张图像中最多检测框的数量 三种 1,10,100
    '''

    x = np.arange(0.0, 1.01, 0.01)
    plt.title(f'{cats["name"]} PR curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.grid(True)

    pr_array1 = cocoEval.eval['precision'][0, :, cats["id"] - 1, 0, 2]
    pr_array2 = cocoEval.eval['precision'][2, :, cats["id"] - 1, 0, 2]
    pr_array3 = cocoEval.eval['precision'][4, :, cats["id"] - 1, 0, 2]

    plt.plot(x, pr_array1, 'b-', label='IoU=0.5')
    plt.plot(x, pr_array2, 'c-', label='IoU=0.6')
    plt.plot(x, pr_array3, 'y-', label='IoU=0.7')

    plt.legend(loc='lower left')
    plt.savefig(f'{cats["name"]}.png')
    plt.close()


def find_bad_match(
        imgRoot,
        cocoGt,
        cocoDt,
        cocoEval,
        cats,
        saveRoot,
):
    imgIds = cocoGt.getImgIds()
    for idx, imgId in enumerate(imgIds):
        bad = False
        badStr = ''
        for cat in cats:
            annoGtIds = cocoGt.getAnnIds(imgIds=[imgId], catIds=[cat['id']])
            annoDtIds = cocoDt.getAnnIds(imgIds=[imgId], catIds=[cat['id']])

            if len(annoGtIds) != len(annoDtIds):
                bad = True
                badStr += '+Miss'

            iou = cocoEval.ious[(imgId, cat['id'])]
            if len(iou) > 0:
                ids = np.where((iou < 0.5) & (iou > 0.1))
                if len(ids[0]) > 0:
                    bad = True
                    ious = '_'.join(list(map(lambda x: f'{x:.2f}', iou[ids])))
                    badStr += f'+IoU{ious}'

        if bad:
            annoGtIds = cocoGt.getAnnIds(imgIds=[imgId])
            annoDtIds = cocoDt.getAnnIds(imgIds=[imgId])
            annoGt = cocoGt.loadAnns(annoGtIds)
            annoDt = cocoDt.loadAnns(annoDtIds)
            imgInfo = cocoGt.loadImgs(ids=[imgId])[0]
            image = cv2.imread(os.path.join(imgRoot, imgInfo['file_name']))
            drawed = drawV2(image, annoGt, annoDt, cats, badStr)
            cv2.imwrite(os.path.join(saveRoot, imgInfo['file_name']), drawed)
            # print(imgInfo)
            print(f'[{idx}/{len(imgIds)}]')


if __name__ == '__main__':
    # imgRoot = '/10t/liubing/CenterNet-master_ip_win/data/images/val'
    # cocoGt = COCO('/10t/liubing/CenterNet-master_ip_win/data/annotations/test.json')
    # cocoDt = cocoGt.loadRes('/10t/XPC/plate_win_r101_result/coco_instances_results_plate_window_test.json')
    # saveRoot = '/10t/XPC/data/PlateWinInfer/valid'

    # imgRoot = '/10t/liubing/CenterNet-master_ip_win/data/images/train'
    # cocoGt = COCO('/10t/liubing/CenterNet-master_ip_win/data/annotations/train_2020.json')
    # cocoDt = cocoGt.loadRes('/10t/XPC/plate_win_r101_result/coco_instances_results_plate_window_train.json')
    # saveRoot = '/10t/XPC/data/PlateWinInfer/train2'

    # imgRoot = '/10t/XPC/data/PlateWinCoco/valid256'
    # cocoGt = COCO('/10t/XPC/data/PlateWinCoco/valid256.json')
    # cocoDt = cocoGt.loadRes('/home/xupeichao/ws/centernet_pro_max2/checkpoints/inference/coco_instances_results.json')
    # saveRoot = '/10t/XPC/data/PlateWinCoco/valid256_analysis'

    imgRoot = '/10t/XPC/data/PlateWinCoco/valid256'
    cocoGt = COCO('/10t/XPC/data/bigPlate/train_2th_1cls.json')
    cocoDt = cocoGt.loadRes('/10t/XPC/outputs/centerbp/1/inference/coco_instances_results.json')
    saveRoot = '/10t/XPC/data/PlateWinCoco/valid256_analysis'

    cats = cocoGt.loadCats(cocoGt.getCatIds())
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # find_bad_match(imgRoot, cocoGt, cocoDt, cocoEval, cats, saveRoot)
    # for i in range(len(cats)):
    #     evaluate_predictions_on_coco(cocoEval, cats[i])
