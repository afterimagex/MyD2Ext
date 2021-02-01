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


from typing import Dict, Tuple

import torch
from torch import Tensor

from detectron2.modeling import detector_postprocess
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class RetinaNetKD(RetinaNet):

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
            in inference, the standard output format, described in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

            if self.teachers:
                teachers_features = self.teachers_forward(batched_inputs)
                losses = self.kd_losses(losses, features, teachers_features)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.no_grad()
    def teachers_forward(self, batched_inputs):
        tea_feature = []
        for teacher in self.teachers:
            images = teacher.preprocess_image(batched_inputs)
            features = teacher.backbone(images.tensor)
            features = [features[f] for f in teacher.head_in_features]
            tea_feature.append(features)
        return tea_feature

    def kd_losses(self, losses, student_features, teachers_features):

        for i, teacher_features in enumerate(teachers_features):
            loss = 0
            for j in range(len(student_features)):
                fea = teacher_features[j].to(student_features[j].device)
                loss += torch.nn.functional.kl_div(student_features[j], fea)
            losses.update({f'kd_loss_{i}': loss * 0.01})

        return losses


@META_ARCH_REGISTRY.register()
class _RetinaNet(RetinaNet):

    def forward(self, images: Tensor):
        # images = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(images)
        features = [features[f] for f in self.head_in_features]
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        return pred_logits, pred_anchor_deltas

    def write_priors(self, images: Tensor, output_priors: str):
        features = self.backbone(images)
        features = [features[f] for f in self.head_in_features]
        anchors = Boxes.cat(self.anchor_generator(features)).tensor.detach().cpu().numpy()

        with open(output_priors, "wb") as f:
            import struct

            shape = anchors.shape
            f.write(struct.pack("=i", len(shape)))
            f.write(struct.pack("={}".format("i" * len(shape)), *shape))
            data = anchors.reshape([-1])
            for d in data:
                f.write(struct.pack("=f", d))
