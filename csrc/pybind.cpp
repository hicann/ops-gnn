/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "host/add_sample/add_sample.h"
#include "host/gather_coo/gather_coo.h"
#include "host/random_walk/random_walk.h"
#include "host/segment_max_csr/segment_max_csr.h"
#include "host/graclus_cluster/graclus_cluster.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ops_gnn: NPU extension";
    m.def("random_walk", &random_walk_npu,
          py::arg("rowptr"), py::arg("col"), py::arg("start"), py::arg("walk_length"),
          py::arg("p") = 1.0, py::arg("q") = 1.0, py::arg("return_edge_indices") = false,
          py::arg("neighbors_sorted") = true,
          "Random walk on a CSR graph (NPU)");
    m.def("add_sample", &add_sample, py::arg("src1"), py::arg("src2"), "两个tensor逐元素相加(NPU)");
    m.def("segment_max_csr", &segment_max_csr, py::arg("src"), py::arg("indptr"),
          py::arg("optional_out") = torch::Tensor(), "Segment max csr(NPU)");
    m.def("graclus_cluster_npu", &graclus_cluster_npu, py::arg("rowptr"), py::arg("col"),
          py::arg("weight"), py::arg("node_perm"), py::arg("num_nodes"), py::arg("has_weight"),
          py::arg("weight_mode"), "Graclus greedy clustering core(NPU)");
    m.def(
        "gather_coo",
        [](torch::Tensor src, torch::Tensor index, py::object outObject) {
            c10::optional<torch::Tensor> optionalOut = c10::nullopt;
            if (!outObject.is_none()) {
                optionalOut = outObject.cast<torch::Tensor>();
            }
            return gather_coo(src, index, optionalOut);
        },
        py::arg("src"),
        py::arg("index"),
        py::arg("out") = py::none(),
        "Gather rows using a COO index on the current NPU stream");
}
