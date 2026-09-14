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
#ifdef OPSGNN_DAV_2201
#include "spmm_max/op_host/spmm_max.h"
#else
#include "gather_coo/op_host/gather_coo.h"
#include "gather_csr/op_host/gather_csr.h"
#include "random_walk/op_host/random_walk.h"
#include "segment_max_csr/op_host/segment_max_csr.h"
#include "radius/op_host/radius.h"
#include "graclus_cluster/op_host/graclus_cluster.h"
#include "ind2ptr/op_host/ind2ptr.h"
#include "ptr2ind/op_host/ptr2ind.h"
#include "scatter/op_host/scatter.h"
#endif

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ops_gnn: NPU extension";
#ifdef OPSGNN_DAV_2201
    m.attr("npu_arch") = "dav-2201";
    m.def("spmm_max_csr", &SpmmMaxCsr, py::arg("indptr"), py::arg("indices"),
          py::arg("x"), py::arg("out") = py::none(), "CSR SpMM max on Ascend NPU");
#else
    m.attr("npu_arch") = "dav-3510";
    m.def("random_walk", &random_walk_npu,
          py::arg("rowptr"), py::arg("col"), py::arg("start"), py::arg("walk_length"),
          py::arg("p") = 1.0, py::arg("q") = 1.0, py::arg("return_edge_indices") = false,
          py::arg("neighbors_sorted") = true,
          "Random walk on a CSR graph (NPU)");
    m.def("gather_csr", &gather_csr, py::arg("src"), py::arg("indptr"),
          py::arg("out") = py::none(), "Gather CSR (NPU)");
    m.def("segment_max_csr", &segment_max_csr, py::arg("src"), py::arg("indptr"),
          py::arg("optional_out") = torch::Tensor(), "Segment max csr(NPU)");
    m.def("radius", &radius_npu, py::arg("x"), py::arg("y"),
          py::arg("ptr_x") = py::none(), py::arg("ptr_y") = py::none(),
          py::arg("r"), py::arg("max_num_neighbors"),
          py::arg("num_workers"), py::arg("ignore_same_index") = false,
          py::arg("stream_handle") = 0,
          "Radius neighbor search on NPU");
    m.def("graclus_cluster_npu", &graclus_cluster_npu, py::arg("rowptr"), py::arg("col"),
          py::arg("weight"), py::arg("node_perm"), py::arg("num_nodes"), py::arg("has_weight"),
          py::arg("weight_mode"), "Graclus greedy clustering core(NPU)");
    m.def("ind2ptr", &ind2ptr, py::arg("ind"), py::arg("M"),
          "Convert sorted row indices to CSR row pointer (NPU), same as torch_sparse.ind2ptr");
    m.def("ptr2ind", &ptr2ind, py::arg("ptr"), py::arg("E"),
          "Convert CSR row pointer to row indices (NPU), same as torch_sparse.ptr2ind");
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
    m.def("scatter_forward", &scatter_forward, py::arg("src"), py::arg("index"),
          py::arg("dim"), py::arg("out"), py::arg("reduce"), py::arg("has_out"),
          py::arg("hot_target") = -1, "torch_scatter-compatible forward reduction(NPU)");
#endif
}
