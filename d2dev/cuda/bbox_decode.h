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

    namespace retinanet {

        int decode(
                int batch_size,
                const void *const *inputs, void **outputs,
                size_t num_anchors, size_t num_classes,
                const std::vector<float> &anchors, const std::vector<float> &weights, float score_thresh, int top_n,
                void *workspace, size_t workspace_size, cudaStream_t stream
        );

        // Interface for Python
        // inline is needed to prevent multiple function definitions when this header is
        // included by different cpps
        inline std::vector <at::Tensor>
        py_decode(at::Tensor cls_head, at::Tensor box_head, std::vector<float> &anchors, std::vector<float> &weights,
                  float score_thresh, int top_n) {
            CHECK_INPUT(cls_head);
            CHECK_INPUT(box_head);

            int batch_size = cls_head.size(0);
            int num_anchors = cls_head.size(1);
            int num_classes = cls_head.size(2);
            auto options = cls_head.options();

            auto scores = at::zeros({batch_size, top_n}, options);
            auto boxes = at::zeros({batch_size, top_n, 4}, options);
            auto classes = at::zeros({batch_size, top_n}, options);

            // Create scratch buffer
            int size = decode(batch_size, nullptr, nullptr, num_anchors, num_classes, anchors, weights, score_thresh,
                              top_n, nullptr, 0, nullptr);
            auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

            // Decode boxes
            std::vector<void *> inputs = {cls_head.data_ptr(), box_head.data_ptr()};
            std::vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr()};

            decode(batch_size, inputs.data(), outputs.data(), num_anchors, num_classes, anchors, weights, score_thresh,
                   top_n, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

            return {scores, boxes, classes};
        }
    }

    namespace retinaplate {
        int decode(
                int batch_size,
                const void *const *inputs, void **outputs,
                size_t num_anchors, size_t num_classes,
                const std::vector<float> &anchors,
                const std::vector<float> &box_weights,
                const std::vector<float> &oks_weights,
                float score_thresh,
                int top_n,
                void *workspace, size_t workspace_size, cudaStream_t stream
        );

        inline std::vector <at::Tensor> py_decode(
                at::Tensor cls_head,
                at::Tensor box_head,
                at::Tensor oks_head,
                std::vector<float> &anchors,
                std::vector<float> &box_weights,
                std::vector<float> &oks_weights,
                float score_thresh, int top_n) {

            CHECK_INPUT(cls_head);
            CHECK_INPUT(box_head);
            CHECK_INPUT(oks_head);

            int batch_size = cls_head.size(0);
            int num_anchors = cls_head.size(1);
            int num_classes = cls_head.size(2);
            auto options = cls_head.options();

            auto scores = at::zeros({batch_size, top_n}, options);
            auto boxes = at::zeros({batch_size, top_n, 4}, options);
            auto classes = at::zeros({batch_size, top_n}, options);
            auto keypoints = at::zeros({batch_size, top_n, 8}, options);

            // Create scratch buffer
            int size = decode(batch_size, nullptr, nullptr, num_anchors, num_classes, anchors, box_weights, oks_weights,
                              score_thresh, top_n, nullptr, 0, nullptr);
            auto scratch = at::zeros({size}, options.dtype(torch::kUInt8));

            // Decode boxes
            std::vector<void *> inputs = {cls_head.data_ptr(), box_head.data_ptr(), oks_head.data_ptr()};
            std::vector<void *> outputs = {scores.data_ptr(), boxes.data_ptr(), classes.data_ptr(),
                                           keypoints.data_ptr()};

            decode(batch_size, inputs.data(), outputs.data(), num_anchors, num_classes, anchors, box_weights,
                   oks_weights, score_thresh, top_n, scratch.data_ptr(), size, at::cuda::getCurrentCUDAStream());

            return {scores, boxes, classes, keypoints};
        }
    }
}