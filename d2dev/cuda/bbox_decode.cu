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

#include "bbox_decode.h"
#include "utils.h"

#include <algorithm>
#include <cstdint>

//  #include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>

namespace d2dev {
    namespace retinanet {


        int decode(int batch_size,
                   const void *const *inputs, void **outputs,
                   size_t num_anchors, size_t num_classes,
                   const std::vector<float> &anchors, const std::vector<float> &weights, float score_thresh, int top_n,
                   void *workspace, size_t workspace_size, cudaStream_t stream) {

            int scores_size = num_anchors * num_classes;

            if (!workspace || !workspace_size)
            {
                // Return required scratch space size cub style
                workspace_size = get_size_aligned<float>(anchors.size());  // anchors
                workspace_size += get_size_aligned<float>(weights.size());
                workspace_size += get_size_aligned<bool>(scores_size);     // flags
                workspace_size += get_size_aligned<int>(scores_size);      // indices
                workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
                workspace_size += get_size_aligned<float>(scores_size);    // scores
                workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

                size_t temp_size_flag = 0;
                thrust::cuda_cub::cub::DeviceSelect::Flagged((void *) nullptr, temp_size_flag,
                                                             thrust::cuda_cub::cub::CountingInputIterator<int>(
                                                                     scores_size),
                                                             (bool *) nullptr, (int *) nullptr, (int *) nullptr,
                                                             scores_size);
                size_t temp_size_sort = 0;
                thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *) nullptr, temp_size_sort,
                                                                            (float *) nullptr, (float *) nullptr,
                                                                            (int *) nullptr, (int *) nullptr,
                                                                            scores_size);
                workspace_size += std::max(temp_size_flag, temp_size_sort);

                return workspace_size;
            }

            auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
            cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice,
                            stream);

            auto weights_d = get_next_ptr<float>(weights.size(), workspace, workspace_size);
            cudaMemcpyAsync(weights_d, weights.data(), weights.size() * sizeof *weights_d, cudaMemcpyHostToDevice,
                            stream);

            auto on_stream = thrust::cuda::par.on(stream);

            auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
            auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
            auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
            auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
            auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

            for (int batch = 0; batch < batch_size; batch++)
            {
                auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
                auto in_boxes = static_cast<const float *>(inputs[1]) + batch * scores_size * 4;

                auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
                auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * top_n;
                auto out_classes = static_cast<float *>(outputs[2]) + batch * top_n;

                // Discard scores below threshold
                thrust::transform(on_stream, in_scores, in_scores + scores_size, flags,
                                  thrust::placeholders::_1 > score_thresh);

                int *num_selected = reinterpret_cast<int *>(indices_sorted);
                thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size,
                                                             thrust::cuda_cub::cub::CountingInputIterator<int>(0),
                                                             flags, indices, num_selected, scores_size, stream);
                cudaStreamSynchronize(stream);
                int num_detections = *thrust::device_pointer_cast(num_selected);

                // Only keep top n scores
                auto indices_filtered = indices;
                if (num_detections > top_n)
                {
                    thrust::gather(on_stream, indices, indices + num_detections, in_scores, scores);
                    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                                                                                scores, scores_sorted, indices,
                                                                                indices_sorted, num_detections, 0,
                                                                                sizeof(*scores) * 8, stream);
                    indices_filtered = indices_sorted;
                    num_detections = top_n;
                }

                // Gather boxes
                thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
                                  thrust::make_zip_iterator(thrust::make_tuple(out_scores, out_boxes, out_classes)),
                                  [=] __device__(int i) {
                                      int idx = i / num_classes;
                                      int cls = i % num_classes;
                                      //    printf("idx: %d\n", idx);

                                      float aw = anchors_d[idx * 4 + 2] - anchors_d[idx * 4 + 0];
                                      float ah = anchors_d[idx * 4 + 3] - anchors_d[idx * 4 + 1];
                                      float ax = anchors_d[idx * 4 + 0] + 0.5 * aw;
                                      float ay = anchors_d[idx * 4 + 1] + 0.5 * ah;
                                      //    printf("aw,ah,ax,ay: %f,%f,%f,%f\n", aw, ah, ax, ay);

                                      float dx = in_boxes[idx * 4 + 0] / weights_d[0];
                                      float dy = in_boxes[idx * 4 + 1] / weights_d[1];
                                      float dw = in_boxes[idx * 4 + 2] / weights_d[2];
                                      float dh = in_boxes[idx * 4 + 3] / weights_d[3];
                                      //    printf("dx,dy,dw,dh: %f,%f,%f,%f\n", dx, dy, dw, dh);

                                      float pred_ctr_x = ax + dx * aw;
                                      float pred_ctr_y = ay + dy * ah;
                                      float pred_w = exp(dw) * aw;
                                      float pred_h = exp(dh) * ah;
                                      //    printf("ctr_x,ctr_y,pred_w,pred_h: %f,%f,%f,%f\n", pred_ctr_x, pred_ctr_y, pred_w, pred_h);

                                      float4 box = float4{
                                              pred_ctr_x - 0.5f * pred_w,
                                              pred_ctr_y - 0.5f * pred_h,
                                              pred_ctr_x + 0.5f * pred_w,
                                              pred_ctr_y + 0.5f * pred_h,
                                      };

                                      return thrust::make_tuple(in_scores[i], box, cls);
                                  });

                // Zero-out unused scores
                if (num_detections < top_n)
                {
                    thrust::fill(on_stream, out_scores + num_detections, out_scores + top_n, 0.0f);
                    thrust::fill(on_stream, out_classes + num_detections, out_classes + top_n, 0.0f);
                }
            }
            return 0;
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
        ) {

            int scores_size = num_anchors * num_classes;

            if (!workspace || !workspace_size)
            {
                // Return required scratch space size cub style
                workspace_size = get_size_aligned<float>(anchors.size());  // anchors
                workspace_size += get_size_aligned<float>(box_weights.size());
                workspace_size += get_size_aligned<float>(oks_weights.size());
                workspace_size += get_size_aligned<bool>(scores_size);     // flags
                workspace_size += get_size_aligned<int>(scores_size);      // indices
                workspace_size += get_size_aligned<int>(scores_size);      // indices_sorted
                workspace_size += get_size_aligned<float>(scores_size);    // scores
                workspace_size += get_size_aligned<float>(scores_size);    // scores_sorted

                size_t temp_size_flag = 0;
                thrust::cuda_cub::cub::DeviceSelect::Flagged((void *) nullptr, temp_size_flag,
                                                             thrust::cuda_cub::cub::CountingInputIterator<int>(
                                                                     scores_size),
                                                             (bool *) nullptr, (int *) nullptr, (int *) nullptr,
                                                             scores_size);
                size_t temp_size_sort = 0;
                thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending((void *) nullptr, temp_size_sort,
                                                                            (float *) nullptr, (float *) nullptr,
                                                                            (int *) nullptr, (int *) nullptr,
                                                                            scores_size);
                workspace_size += std::max(temp_size_flag, temp_size_sort);

                return workspace_size;
            }

            auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
            cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice,
                            stream);

            auto box_weights_d = get_next_ptr<float>(box_weights.size(), workspace, workspace_size);
            cudaMemcpyAsync(box_weights_d, box_weights.data(), box_weights.size() * sizeof *box_weights_d,
                            cudaMemcpyHostToDevice, stream);

            auto oks_weights_d = get_next_ptr<float>(oks_weights.size(), workspace, workspace_size);
            cudaMemcpyAsync(oks_weights_d, oks_weights.data(), oks_weights.size() * sizeof *oks_weights_d,
                            cudaMemcpyHostToDevice, stream);

            auto on_stream = thrust::cuda::par.on(stream);
            auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
            auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
            auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
            auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
            auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);

            for (int batch = 0; batch < batch_size; batch++)
            {
                auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
                auto in_boxes = static_cast<const float *>(inputs[1]) + batch * scores_size * 4;
                auto in_keypoints = static_cast<const float *>(inputs[2]) + batch * scores_size * 8;

                auto out_scores = static_cast<float *>(outputs[0]) + batch * top_n;
                auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * top_n;
                auto out_classes = static_cast<float *>(outputs[2]) + batch * top_n;
                auto out_keypoints = static_cast<float8 *>(outputs[3]) + batch * top_n;

                // Discard scores below threshold
                thrust::transform(on_stream, in_scores, in_scores + scores_size, flags,
                                  thrust::placeholders::_1 > score_thresh);

                int *num_selected = reinterpret_cast<int *>(indices_sorted);
                thrust::cuda_cub::cub::DeviceSelect::Flagged(workspace, workspace_size,
                                                             thrust::cuda_cub::cub::CountingInputIterator<int>(0),
                                                             flags, indices, num_selected, scores_size, stream);
                cudaStreamSynchronize(stream);
                int num_detections = *thrust::device_pointer_cast(num_selected);

                // Only keep top n scores
                auto indices_filtered = indices;
                if (num_detections > top_n)
                {
                    thrust::gather(on_stream, indices, indices + num_detections, in_scores, scores);
                    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
                                                                                scores, scores_sorted, indices,
                                                                                indices_sorted, num_detections, 0,
                                                                                sizeof(*scores) * 8, stream);
                    indices_filtered = indices_sorted;
                    num_detections = top_n;
                }

                // Gather boxes
                thrust::transform(on_stream, indices_filtered, indices_filtered + num_detections,
                                  thrust::make_zip_iterator(
                                          thrust::make_tuple(out_scores, out_boxes, out_classes, out_keypoints)),
                                  [=] __device__(int i) {
                                      int idx = i / num_classes;
                                      int cls = i % num_classes;
                                      //    printf("idx: %d\n", idx);

                                      float aw = anchors_d[idx * 4 + 2] - anchors_d[idx * 4 + 0];
                                      float ah = anchors_d[idx * 4 + 3] - anchors_d[idx * 4 + 1];
                                      float ax = anchors_d[idx * 4 + 0] + 0.5 * aw;
                                      float ay = anchors_d[idx * 4 + 1] + 0.5 * ah;
                                      //    printf("aw,ah,ax,ay: %f,%f,%f,%f\n", aw, ah, ax, ay);

                                      float dx = in_boxes[idx * 4 + 0] / box_weights_d[0];
                                      float dy = in_boxes[idx * 4 + 1] / box_weights_d[1];
                                      float dw = in_boxes[idx * 4 + 2] / box_weights_d[2];
                                      float dh = in_boxes[idx * 4 + 3] / box_weights_d[3];
                                      //    printf("dx,dy,dw,dh: %f,%f,%f,%f\n", dx, dy, dw, dh);

                                      float pred_ctr_x = ax + dx * aw;
                                      float pred_ctr_y = ay + dy * ah;
                                      float pred_w = exp(dw) * aw;
                                      float pred_h = exp(dh) * ah;
                                      //    printf("ctr_x,ctr_y,pred_w,pred_h: %f,%f,%f,%f\n", pred_ctr_x, pred_ctr_y, pred_w, pred_h);

                                      float4 box = float4{
                                              pred_ctr_x - 0.5f * pred_w,
                                              pred_ctr_y - 0.5f * pred_h,
                                              pred_ctr_x + 0.5f * pred_w,
                                              pred_ctr_y + 0.5f * pred_h,
                                      };

                                      float8 pts = float8(
                                              ax + aw * in_keypoints[idx * 8 + 0] / oks_weights_d[0],
                                              ay + ah * in_keypoints[idx * 8 + 1] / oks_weights_d[1],
                                              ax + aw * in_keypoints[idx * 8 + 2] / oks_weights_d[2],
                                              ay + ah * in_keypoints[idx * 8 + 3] / oks_weights_d[3],
                                              ax + aw * in_keypoints[idx * 8 + 4] / oks_weights_d[4],
                                              ay + ah * in_keypoints[idx * 8 + 5] / oks_weights_d[5],
                                              ax + aw * in_keypoints[idx * 8 + 6] / oks_weights_d[6],
                                              ay + ah * in_keypoints[idx * 8 + 7] / oks_weights_d[7]
                                      );

                                      return thrust::make_tuple(in_scores[i], box, cls, pts);
                                  });

                // Zero-out unused scores
                if (num_detections < top_n)
                {
                    thrust::fill(on_stream, out_scores + num_detections, out_scores + top_n, 0.0f);
                    thrust::fill(on_stream, out_classes + num_detections, out_classes + top_n, 0.0f);
                    thrust::fill(on_stream, out_keypoints + num_detections, out_keypoints + top_n, 0.0f);
                }
            }

            return 0;
        }
    }
}