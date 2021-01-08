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


import numpy as np
import torch

__all__ = ["CenterNetGT"]


class CenterNetGT(object):
    def __init__(self, num_classes, down_scale, min_overlap, tensor_dim):
        self.num_classes = num_classes
        self.down_scale = 1 / down_scale
        self.min_overlap = min_overlap
        self.tensor_dim = tensor_dim

    @torch.no_grad()
    def __call__(self, gt_instances, input_shape):
        hm_list, wh_list, reg_list, index_list, reg_mask_list = [[] for _ in range(5)]
        h, w = input_shape[-2:]
        output_size = [int(h * self.down_scale), int(w * self.down_scale)]

        for gt_per_image in gt_instances:

            gt_hm = torch.zeros(self.num_classes, *output_size)
            gt_wh = torch.zeros(self.tensor_dim, 2)
            gt_reg = torch.zeros_like(gt_wh)
            gt_index = torch.zeros(self.tensor_dim)
            reg_mask = torch.zeros(self.tensor_dim)

            num_boxes = min(len(gt_per_image), self.tensor_dim)

            gt_boxes = gt_per_image.gt_boxes
            gt_boxes.scale(self.down_scale, self.down_scale)

            centers = gt_boxes.get_centers()[:num_boxes]
            centers_int = centers.to(torch.int32)
            gt_index[:num_boxes] = centers_int[:num_boxes, 1] * output_size[1] + centers_int[:num_boxes, 0]
            gt_reg[:num_boxes] = centers[:num_boxes] - centers_int[:num_boxes]
            reg_mask[:num_boxes] = 1

            wh = torch.zeros_like(centers)
            gt_boxes_tensor = gt_boxes.tensor[:num_boxes]
            wh[..., 0] = gt_boxes_tensor[..., 2] - gt_boxes_tensor[..., 0]
            wh[..., 1] = gt_boxes_tensor[..., 3] - gt_boxes_tensor[..., 1]
            self.generate_score_map(gt_per_image.gt_classes[:num_boxes], gt_hm, wh, centers_int)
            gt_wh[:num_boxes] = wh

            hm_list.append(gt_hm)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(reg_mask)
            index_list.append(gt_index)

        return {
            'gt_hm': torch.stack(hm_list, dim=0),
            'gt_wh': torch.stack(wh_list, dim=0),
            'gt_reg': torch.stack(reg_list, dim=0),
            'gt_mask': torch.stack(reg_mask_list, dim=0),
            'gt_index': torch.stack(index_list, dim=0),
        }

    def generate_score_map(self, gt_classes, gt_hm, wh, centers_int):
        radius = self.get_gaussian_radius(wh)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_classes.shape[0]):
            self.draw_gaussian(gt_hm[gt_classes[i]], centers_int[i], radius[i])

    def get_gaussian_radius(self, wh):
        # boxes_tensor = torch.Tensor(wh)
        width, height = wh[..., 0], wh[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - self.min_overlap) / (1 + self.min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - self.min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * self.min_overlap
        b3 = -2 * self.min_overlap * (height + width)
        c3 = (self.min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)

        return torch.min(r1, torch.min(r2, r3))

    def draw_gaussian(self, gt_hm, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = gt_hm.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = gt_hm[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            gt_hm[y - top: y + bottom, x - left: x + right] = masked_fmap

    def gaussian2D(self, radius, sigma=1):

        m, n = radius
        y, x = np.ogrid[-m: m + 1, -n: n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss
