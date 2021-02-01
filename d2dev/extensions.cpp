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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <optional>

#include "cuda/bbox_decode.h"
#include "cuda/nms.h"
#include "engine/retinaplate.h"


namespace d2dev {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<retinaplate::Engine>(m, "Engine")
    .def(pybind11::init<const char *, size_t, size_t, string, float, int, const vector<vector<float>>&, float, int, const vector<string>&, string, string, bool>())
    .def("save", &retinaplate::Engine::save)
    .def("infer", &retinaplate::Engine::infer)
    .def_static("load", [](const string &path) {
        return new retinaplate::Engine(path);
    })
//    .def("__call__", [](retinaplate::Engine &engine, at::Tensor data) {
//        return infer(engine, data);
//    });

    m.def("batch_nms", &py_batch_nms, "batch nms");
    m.def("retinanet_decode", &retinanet::py_decode, "retinanet decode");
    }
}