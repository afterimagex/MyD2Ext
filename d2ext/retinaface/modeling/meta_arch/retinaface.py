# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, get_norm, nonzero_tuple
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone.build import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from fvcore.nn import giou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from torch import Tensor, nn
from torch.nn import functional as F

from ..landmark_regression import Mark2MarkTransform

__all__ = ["RetinaFace"]


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class RetinaFace(nn.Module):
    """
    Implement RetinaNet in :paper:`RetinaNet`.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            head,
            head_in_features,
            anchor_generator,
            box2box_transform,
            mark2mark_transform,
            anchor_matcher,
            num_classes,
            num_landmark,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0,
            smooth_l1_beta=0.1,
            loc_weight=2.0,
            box_reg_loss_type="smooth_l1",
            test_score_thresh=0.05,
            test_topk_candidates=1000,
            test_nms_thresh=0.5,
            max_detections_per_image=100,
            pixel_mean,
            pixel_std,
            vis_period=0,
            input_format="BGR",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            num_classes (int): number of classes. Used to label background proposals.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            smooth_l1_beta (float): smooth_l1_beta
            box_reg_loss_type (str): Options are "smooth_l1", "giou"

            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).

            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
            vis_period (int):
                The period (in terms of steps) for minibatch visualization at train time.
                Set to 0 to disable.
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features

        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.mark2mark_transform = mark2mark_transform
        self.anchor_matcher = anchor_matcher

        self.num_classes = num_classes
        self.num_landmark = num_landmark
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.loc_weight = loc_weight
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": RetinaFaceHead(cfg, feature_shapes),
            "anchor_generator": anchor_generator,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS),
            "mark2mark_transform": Mark2MarkTransform(cfg.MODEL.RETINAFACE.NUM_LANDMARK,
                                                      weights=cfg.MODEL.RETINAFACE.LANDMARK_REG_WEIGHTS),
            "anchor_matcher": Matcher(
                cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                cfg.MODEL.RETINANET.IOU_LABELS,
                allow_low_quality_matches=True,
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "num_landmark": cfg.MODEL.RETINAFACE.NUM_LANDMARK,
            "head_in_features": cfg.MODEL.RETINANET.IN_FEATURES,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE,
            "loc_weight": cfg.MODEL.RETINAFACE.LOC_WEIGHT,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.RETINANET.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
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
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(
            boxes=batched_inputs[image_index]["instances"].gt_boxes,
            keypoints=batched_inputs[image_index]["instances"].gt_keypoints,
        )
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()
        predicted_keypoints = processed_results.pred_keypoints.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(
            boxes=predicted_boxes[0:max_boxes],
            keypoints=predicted_keypoints[0:max_boxes],
        )
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

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
        pred_logits, pred_anchor_deltas, pred_mark_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        pred_mark_deltas = [permute_to_N_HWA_K(x, self.num_landmark * 2) for x in pred_mark_deltas]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes, gt_marks, gt_marks_labels = self.label_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_mark_deltas,
                                 gt_marks, gt_marks_labels)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, pred_mark_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(anchors, pred_logits, pred_anchor_deltas, pred_mark_deltas, images.image_sizes)
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

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_mark_deltas,
               gt_marks, gt_marks_labels):
        """
        Args:
            For `gt_classes`, `gt_anchors_deltas`, `gt_landmarks_deltas` and `gt_landmarks_labels` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R), (N, R, 4), (N, R, num_landmark * 2) and (N, R), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas` and `pred_landmark_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls", "loss_box_reg" and "loss_landmark_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        gt_landmark_deltas = [self.mark2mark_transform.get_deltas(anchors, k) for k in gt_marks]
        gt_landmark_deltas = torch.stack(gt_landmark_deltas)  # (N, R, Marks * 2)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss
        gt_labels_target = F.one_hot(
            gt_labels[valid_mask],
            num_classes=self.num_classes + 1)[:, :-1]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        if self.box_reg_loss_type == "smooth_l1":
            loss_box_reg = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            pred_boxes = [
                self.box2box_transform.apply_deltas(k, anchors)
                for k in cat(pred_anchor_deltas, dim=1)
            ]
            loss_box_reg = giou_loss(
                torch.stack(pred_boxes)[pos_mask], torch.stack(gt_boxes)[pos_mask], reduction="sum"
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg * self.loc_weight
        # landmark regression loss
        # NOTE filter in-valid landmarks
        gt_marks_labels = torch.stack(gt_marks_labels)
        marks_pos_mask = pos_mask & (gt_marks_labels > 0)

        # NOTE loss_normalizer for landmark may be not consistence with score or bbox
        loss_marks_reg = smooth_l1_loss(
            cat(pred_mark_deltas, dim=1)[marks_pos_mask],
            gt_landmark_deltas[marks_pos_mask],
            beta=self.smooth_l1_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer,
            "loss_marks_reg": loss_marks_reg / self.loss_normalizer,
        }

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """
        anchors = Boxes.cat(anchors)  # Rx4
        num_anchors = anchors.tensor.shape[0]

        gt_labels = []
        matched_gt_boxes = []
        matched_gt_marks = []
        matched_gt_marks_labels = []
        for gt_per_image in gt_instances:
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                matched_gt_marks_iv = gt_per_image.gt_keypoints.tensor[matched_idxs]

                matched_gt_marks_i = matched_gt_marks_iv[:, :, :2].flatten(1)
                matched_gt_marks_labels_i = matched_gt_marks_iv[:, :, 2].flatten(1)
                matched_gt_marks_labels_i, _ = torch.min(matched_gt_marks_labels_i, dim=1)

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes
                matched_gt_marks_i = torch.zeros(num_anchors, self.num_landmark * 2).to(self.device)
                matched_gt_marks_labels_i = torch.zeros(num_anchors).to(self.device)

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            matched_gt_marks.append(matched_gt_marks_i)
            matched_gt_marks_labels.append(matched_gt_marks_labels_i)

        return gt_labels, matched_gt_boxes, matched_gt_marks, matched_gt_marks_labels

    def inference(
            self,
            anchors: List[Boxes],
            pred_logits: List[Tensor],
            pred_anchor_deltas: List[Tensor],
            pred_marks_deltas: List[Tensor],
            image_sizes: List[Tuple[int, int]],
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results: List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            bbox_deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            marks_deltas_per_image = [x[img_idx] for x in pred_marks_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, bbox_deltas_per_image, marks_deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
            self,
            anchors: List[Boxes],
            box_cls: List[Tensor],
            box_delta: List[Tensor],
            marks_delta: List[Tensor],
            image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        marks_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, marks_reg_i, anchors_i in zip(box_cls, box_delta, marks_delta, anchors):
            # HxWxAxK,
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            marks_reg_i = marks_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
            predicted_marks = self.mark2mark_transform.apply_deltas(marks_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            marks_all.append(predicted_marks)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, marks_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, marks_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        keypoints_all = marks_all[keep].reshape(-1, self.num_landmark, 2)
        keypoints_all = torch.cat(
            (keypoints_all, 2 * torch.ones(keypoints_all.shape[0], self.num_landmark, 1).to(self.device)), dim=2)
        result.pred_keypoints = keypoints_all  # Keypoints(keypoints_all)

        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class RetinaFaceHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
            self,
            *,
            input_shape: List[ShapeSpec],
            num_classes,
            num_landmark,
            num_anchors,
            conv_dims: List[int],
            norm="",
            use_ssh=True,
            prior_prob=0.01,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
        super().__init__()

        if norm == "BN" or norm == "SyncBN":
            logger = logging.getLogger(__name__)
            logger.warn("Shared norm does not work well for BN, SyncBN, expect poor results")

        cls_subnet = []
        bbox_subnet = []
        marks_subnet = []
        for in_channels, out_channels in zip([input_shape[0].channels] + conv_dims, conv_dims):
            if use_ssh:
                cls_subnet.append(SSH(in_channels, out_channels))
                bbox_subnet.append(SSH(in_channels, out_channels))
                marks_subnet.append(SSH(in_channels, out_channels))
            else:
                cls_subnet.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                if norm:
                    cls_subnet.append(get_norm(norm, out_channels))
                cls_subnet.append(nn.ReLU())
                bbox_subnet.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                if norm:
                    bbox_subnet.append(get_norm(norm, out_channels))
                bbox_subnet.append(nn.ReLU())
                marks_subnet.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                if norm:
                    marks_subnet.append(get_norm(norm, out_channels))
                marks_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.marks_subnet = nn.Sequential(*marks_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        self.marks_pred = nn.Conv2d(
            conv_dims[-1], num_anchors * num_landmark * 2, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet, self.marks_subnet,
            self.cls_score, self.bbox_pred, self.marks_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        assert (
                len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
            "num_landmark": cfg.MODEL.RETINAFACE.NUM_LANDMARK,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.RETINANET.NUM_CONVS,
            "prior_prob": cfg.MODEL.RETINANET.PRIOR_PROB,
            "norm": cfg.MODEL.RETINANET.NORM,
            "use_ssh": cfg.MODEL.RETINAFACE.USE_SSH,
            "num_anchors": num_anchors,
        }

    def forward(self, features: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        marks_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
            marks_reg.append(self.marks_pred(self.marks_subnet(feature)))
        return logits, bbox_reg, marks_reg


class SSH(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0, "SSH Module required out_channel to be n * 4"
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 2)
        )
        self.conv5X5_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=leaky)
        )
        self.conv5X5_2 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.conv7X7_2 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=leaky)
        )
        self.conv7x7_3 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv3X3 = self.conv3X3(x)
        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = self.relu(out)
        return out
