/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <optional>

#include "utils.h"

namespace d2dev {

    int batch_nms(int batchSize,
                  const void *const *inputs, void **outputs,
                  size_t count, int detections_per_im, float nms_thresh,
                  void *workspace, size_t workspace_size, cudaStream_t stream);

    int plate_batch_nms(int batch_size,
                        const void *const *inputs, void **outputs,
                        size_t count, int detections_per_im, float nms_thresh,
                        void *workspace, size_t workspace_size, cudaStream_t stream);

    // Interface for Python
    // inline is needed to prevent multiple function definitions when this header is
    // included by different cpps
    inline std::vector <at::Tensor> py_batch_nms(at::Tensor scores, at::Tensor boxes, at::Tensor classes,
                                                 float nms_thresh, int detections_per_im) {

        CHECK_INPUT(scores);
        CHECK_INPUT(boxes);
        CHECK_INPUT(classes);

        int batch = scores.size(0);
        int count = scores.size(1);
        auto options = scores.options();

        auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
        auto nms_boxes = at::zeros({batch, detections_per_im, 4}, boxes.options());
        auto nms_classes = at::zeros({batch, detections_per_im}, classes.options());

        // Create scratch buffer
        int size = batch_nms(batch, nullptr, nullptr, count,
                             detections_per_im, nms_thresh, nullptr, 0, nullptr);
        auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

        // Perform NMS
        std::vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};
        std::vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr(), nms_classes.data_ptr()};
        batch_nms(batch, inputs.data(), outputs.data(), count,
                  detections_per_im, nms_thresh,
                  scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

        return {nms_scores, nms_boxes, nms_classes};
    }

    inline std::vector <at::Tensor>
    py_plate_batch_nms(at::Tensor scores, at::Tensor boxes, at::Tensor classes, at::Tensor keypoints,
                       float nms_thresh, int detections_per_im) {

        CHECK_INPUT(scores);
        CHECK_INPUT(boxes);
        CHECK_INPUT(classes);
        CHECK_INPUT(keypoints);

        int batch = scores.size(0);
        int count = scores.size(1);
        auto options = scores.options();

        auto nms_scores = at::zeros({batch, detections_per_im}, scores.options());
        auto nms_boxes = at::zeros({batch, detections_per_im, 4}, boxes.options());
        auto nms_classes = at::zeros({batch, detections_per_im}, classes.options());
        auto nms_keypoints = at::zeros({batch, detections_per_im, 8}, classes.options());

        // Create scratch buffer
        int size = batch_nms(batch, nullptr, nullptr, count,
                             detections_per_im, nms_thresh, nullptr, 0, nullptr);
        auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

        // Perform NMS
        std::vector<void *> inputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr(), keypoints.data_ptr()};
        std::vector<void *> outputs = {nms_scores.data_ptr(), nms_boxes.data_ptr(), nms_classes.data_ptr(),
                                       nms_keypoints.data_ptr()};
        plate_batch_nms(batch, inputs.data(), outputs.data(), count,
                        detections_per_im, nms_thresh,
                        scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

        return {nms_scores, nms_boxes, nms_classes, nms_keypoints};
    }
}