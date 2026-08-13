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
#include "host/add_sample/add_sample.h"
#include "host/segment_max_csr/segment_max_csr.h"
#include "host/graclus_cluster/graclus_cluster.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ops_gnn: NPU extension";
    m.def("add_sample", &add_sample, py::arg("src1"), py::arg("src2"), "两个tensor逐元素相加(NPU)");
    m.def("segment_max_csr", &segment_max_csr, py::arg("src"), py::arg("indptr"),
          py::arg("optional_out") = torch::Tensor(), "Segment max csr(NPU)");
    m.def("graclus_cluster_npu", &graclus_cluster_npu, py::arg("rowptr"), py::arg("col"),
          py::arg("weight"), py::arg("node_perm"), py::arg("num_nodes"), py::arg("has_weight"),
          py::arg("weight_mode"), "Graclus greedy clustering core(NPU)");
}
