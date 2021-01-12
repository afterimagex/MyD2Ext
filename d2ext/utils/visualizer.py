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


from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode


class TrainingVisualizer:
    def __init__(self, postprocess=None, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        self.metadata = metadata
        self.postprocess = postprocess
        self.scale = scale
        self.instance_mode = instance_mode

    def draw_instance_groundtruth(self, img_rgb, groundtruth):
        visual = Visualizer(img_rgb, self.metadata, self.scale, self.instance_mode)
        instance = Instances(
            image_size=groundtruth.image_size,
            pred_boxes=groundtruth.gt_boxes,
            pred_classes=groundtruth.gt_classes,
        )
        if groundtruth.has('gt_keypoints'):
            instance.pred_keypoints = groundtruth.gt_keypoints
        vis_img = visual.draw_instance_predictions(instance)
        return vis_img.get_image()

    def draw_instance_predictions(self, img_rgb, predictions, num_instance=20):
        processed_results = self.postprocess(predictions, img_rgb.shape[0], img_rgb.shape[1])
        visual = Visualizer(img_rgb, self.metadata, self.scale, self.instance_mode)
        instance = Instances(
            image_size=processed_results.image_size,
            pred_boxes=processed_results.pred_boxes.tensor.detach().cpu().numpy()[:num_instance],
            scores=processed_results.scores.detach().cpu().numpy()[:num_instance],
            pred_classes=processed_results.pred_classes.detach().cpu().int().numpy()[:num_instance],
        )
        if processed_results.has('pred_keypoints'):
            instance.pred_keypoints = processed_results.pred_keypoints.detach().cpu().numpy()[:num_instance]

        vis_img = visual.draw_instance_predictions(instance)
        return vis_img.get_image()
