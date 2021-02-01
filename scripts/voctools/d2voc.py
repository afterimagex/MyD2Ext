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

import torch
import atexit
import bisect
import multiprocessing as mp

import torch.backends.cudnn as cudnn

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from pascal_voc_writer import Writer

cudnn.benchmark = True



class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(
                gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(
                    cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()

    cfg.merge_from_file(
        '/home/xupeichao/ws/AutoTrain/trainunify/contribute/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = '/10t/XPC/outputs/d2out/fdh/0/model_final.pth'
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def walk_dirs(paths, suffix):
    dir_map = []

    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        for (root, dirs, files) in os.walk(path):
            for item in files:
                if item.endswith(suffix):
                    d = os.path.abspath(os.path.join(root, item))
                    dir_map.append(d)
    return dir_map


def draw_result(image, boxes, classes, save_path):
    default_class = ['big_iPlateRect']

    for i, box in enumerate(boxes):
        name = default_class[classes[i]]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.rectangle(image, (x1, y1 - 20), (x2, y1), (255, 255, 0), -1)
        cv2.putText(image, name, (x1, y1), 1, 1, (0, 0, 0))
    cv2.imwrite(save_path, image)


def main():
    mp.set_start_method("spawn", force=True)
    num_gpu = torch.cuda.device_count()

    cfg = setup_cfg()
    predictor = AsyncPredictor(cfg, num_gpus=num_gpu)

    save_dir = '/10t/XPC/data/big_iplaterect_infer_xml'
    default_class = ['big_iPlateRect']

    for imfile in walk_dirs('/10t/XPC/data/big_iplaterect', 'jpg'):
        image = cv2.imread(imfile)
        instances = predictor(image)['instances']
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        # scores = instances.scores.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()

        if len(pred_classes) > 0:

            draw_result(image, pred_boxes, pred_classes, os.path.join(save_dir, os.path.split(imfile)[-1]))

            writer = Writer(
                path=imfile,
                width=image.shape[1],
                height=image.shape[0],
                database='big_iPlateRect',
            )
            for i, boxes in enumerate(pred_boxes):
                name = default_class[pred_classes[i]]
                boxes = list(map(int, boxes))
                writer.addObject(name, boxes[0], boxes[1], boxes[2], boxes[3])
            writer.save(os.path.join(save_dir, os.path.split(imfile)[-1].replace('.jpg', '.xml')))

        print(imfile)


if __name__ == '__main__':
    main()
