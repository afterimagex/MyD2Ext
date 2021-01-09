# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from d2ext.layers.centernet_deconv import DeconvLayer
from d2ext.layers.centernet_loss import reg_l1_loss, modified_focal_loss, ignore_unlabel_focal_loss, mse_loss
from d2ext.layers.utils import pseudo_nms, topk_score, gather_feature
from d2ext.modeling.centernet_gt import CenterNetGT
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage

__all__ = ["CenterNet"]


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            upsample,
            head,
            num_classes,
            gt_generator,
            pixel_mean,
            pixel_std,
            max_detections_per_image=100,
            classification_loss_type,
            regression_loss_type,
            head_in_features,
            test_score_thresh,
            hm_norm,
            hm_weight,
            wh_weight,
            reg_weight,
            kd_enable=False,
            kd_loss,
            kd_without_label,
            metadata,
            vis_period=0,
            input_format="BGR",
    ):
        super().__init__()

        self.backbone = backbone
        self.upsample = upsample
        self.head = head
        self.head_in_features = head_in_features

        self.num_classes = num_classes
        self.gt_generator = gt_generator

        # Inference parameters:
        self.max_detections_per_image = max_detections_per_image
        self.test_score_thresh = test_score_thresh

        # Loss parameters:
        self.classification_loss_type = classification_loss_type
        self.regression_loss_type = regression_loss_type

        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format
        self.metadata = metadata

        if kd_enable:
            self.kd_loss = kd_loss
            self.kd_without_label = kd_without_label

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        self.hm_norm = hm_norm
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        feature_shapes = backbone.output_shape()[cfg.MODEL.CENTERNET.IN_FEATURES]
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused"
        )
        return {
            "backbone": backbone,
            "upsample": CenternetDeconv(cfg, feature_shapes),
            "head": CenternetHead(cfg),
            "head_in_features": cfg.MODEL.CENTERNET.IN_FEATURES,
            "num_classes": cfg.MODEL.CENTERNET.NUM_CLASSES,
            "test_score_thresh": cfg.MODEL.CENTERNET.SCORE_THRESH_TEST,
            "gt_generator": CenterNetGT(
                cfg.MODEL.CENTERNET.NUM_CLASSES,
                cfg.MODEL.CENTERNET.DOWN_SCALE,
                cfg.MODEL.CENTERNET.MIN_OVERLAP,
                cfg.MODEL.CENTERNET.TENSOR_DIM,
            ),
            "metadata": metadata,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "classification_loss_type": cfg.MODEL.CENTERNET.LOSS.IGNORE_UNLABEL,
            "regression_loss_type": cfg.MODEL.CENTERNET.LOSS.MSE,
            "hm_norm": cfg.MODEL.CENTERNET.LOSS.NORM_WH,
            "hm_weight": cfg.MODEL.CENTERNET.LOSS.HM_WEIGHT,
            'wh_weight': cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT,
            'reg_weight': cfg.MODEL.CENTERNET.LOSS.REG_WEIGHT,
            "kd_enable": cfg.MODEL.CENTERNET.KD.ENABLED,
            "kd_loss": None,
            "kd_without_label": cfg.MODEL.CENTERNET.KD.KD_WITHOUT_LABEL,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, self.metadata)
        v_gt = v_gt.overlay_instances(
            boxes=batched_inputs[image_index]["instances"].gt_boxes,
            labels=batched_inputs[image_index]["instances"].gt_classes.cpu().numpy(),
        )

        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        processed_instance = processed_results.to(torch.device("cpu"))
        processed_instance.pred_classes = processed_instance.pred_classes.to(torch.int32)
        v_pred = Visualizer(img, self.metadata)
        v_pred = v_pred.draw_instance_predictions(
            predictions=processed_instance
        )
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)[self.head_in_features]
        features = self.upsample(features)
        pred_dict = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_dict = self.gt_generator(gt_instances, images.tensor.shape)

            losses = self.losses(pred_dict, gt_dict)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        pred_dict['pred_hm'], pred_dict['pred_wh'], pred_dict['pred_reg'],
                        images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(pred_dict['pred_hm'], pred_dict['pred_wh'], pred_dict['pred_reg'],
                                     images.image_sizes)
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

    def decode(self, pred_hm, pred_wh, pred_reg, batch_size):
        pred_hm = pseudo_nms(pred_hm)
        scores, index, class_idxs, ys, xs = topk_score(pred_hm, K=self.max_detections_per_image)

        class_idxs = class_idxs.reshape(batch_size, self.max_detections_per_image).float()
        scores = scores.reshape(batch_size, self.max_detections_per_image)

        pred_reg = gather_feature(pred_reg, index, use_transform=True)
        pred_reg = pred_reg.reshape(batch_size, self.max_detections_per_image, 2)
        xs = xs.view(batch_size, self.max_detections_per_image, 1) + pred_reg[:, :, 0:1]
        ys = ys.view(batch_size, self.max_detections_per_image, 1) + pred_reg[:, :, 1:2]

        pred_wh = gather_feature(pred_wh, index, use_transform=True)
        pred_wh = pred_wh.reshape(batch_size, self.max_detections_per_image, 2)

        half_w, half_h = pred_wh[..., 0:1] / 2, pred_wh[..., 1:2] / 2
        boxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2) / self.gt_generator.down_scale

        return boxes, scores, class_idxs

    @torch.no_grad()
    def inference(self, pred_hm, pred_wh, pred_reg, image_sizes: List[Tuple[int, int]]) -> List[Instances]:
        boxes, scores, class_idxs = self.decode(pred_hm, pred_wh, pred_reg, len(image_sizes))
        return [
            Instances(image_size, pred_boxes=Boxes(boxes[i]), scores=scores[i], pred_classes=class_idxs[i])
            for i, image_size in enumerate(image_sizes)
        ]

    def losses(self, pred_dict, gt_dict):
        """
        calculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
                "gt_hm": gt scoremap,
                "gt_wh": gt width and height of boxes,
                "gt_reg": gt regression of box center point,
                "gt_mask": mask of regression,
                "gt_index": gt index,
            }
            pred(dict): a dict contains all information of prediction
            pred = {
                "pred_hm": predicted score map
                "pred_wh": predicted width and height of box
                "pred_reg": predcited regression
            }
        """
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(self.device)

        if self.classification_loss_type == 'ignore_unlabel':
            loss_cls = ignore_unlabel_focal_loss(pred_dict['pred_hm'], gt_dict["gt_hm"])
        elif self.classification_loss_type == 'mse':
            loss_cls = mse_loss(pred_dict['pred_hm'], gt_dict["gt_hm"])
        else:
            loss_cls = modified_focal_loss(pred_dict['pred_hm'], gt_dict["gt_hm"])

        mask = gt_dict["gt_mask"]
        index = gt_dict["gt_index"].to(torch.long)
        # width and height loss, better version

        loss_wh = reg_l1_loss(pred_dict["pred_wh"], mask, index, gt_dict["gt_wh"], self.hm_norm)
        # regression loss
        loss_reg = reg_l1_loss(pred_dict["pred_reg"], mask, index, gt_dict["gt_reg"])

        return {
            "loss_cls": loss_cls * self.hm_weight,
            "loss_box": loss_wh * self.wh_weight,
            "loss_reg": loss_reg * self.reg_weight,
        }

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)
        return images


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
            self,
            *,
            in_channel,
            num_classes,
    ):
        super(CenternetHead, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, num_classes, kernel_size=1),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, 2, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel, 2, kernel_size=1),
        )
        self.cls_head[2].bias.data.fill_(-2.19)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channel": cfg.MODEL.CENTERNET.DECONV.CHANNEL[-1],
            "num_classes": cfg.MODEL.CENTERNET.NUM_CLASSES,
        }

    def forward(self, x):
        return {
            'pred_hm': torch.sigmoid(self.cls_head(x)),
            'pred_wh': self.wh_head(x),
            'pred_reg': self.reg_head(x),
        }


class CenternetDeconv(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
            self,
            *,
            input_shape: ShapeSpec,
            deconv_channel: list,
            deconv_kernel: list,
            deform,
            deform_modulate,
    ):
        super(CenternetDeconv, self).__init__()
        self.deconv1 = DeconvLayer(
            in_channels=input_shape.channels,
            out_channels=deconv_channel[0],
            kernel_size=deconv_kernel[0],
            deform=deform,
            deform_modulate=deform_modulate,
        )
        self.deconv2 = DeconvLayer(
            in_channels=deconv_channel[0],
            out_channels=deconv_channel[1],
            kernel_size=deconv_kernel[1],
            deform=deform,
            deform_modulate=deform_modulate,
        )
        self.deconv3 = DeconvLayer(
            in_channels=deconv_channel[1],
            out_channels=deconv_channel[2],
            kernel_size=deconv_kernel[2],
            deform=deform,
            deform_modulate=deform_modulate,
        )

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        return {
            "input_shape": input_shape,
            "deconv_channel": cfg.MODEL.CENTERNET.DECONV.CHANNEL,
            "deconv_kernel": cfg.MODEL.CENTERNET.DECONV.KERNEL,
            "deform": cfg.MODEL.CENTERNET.DECONV.DEFORM,
            "deform_modulate": cfg.MODEL.CENTERNET.DECONV.DEFORM_MODULATED,
        }

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x
