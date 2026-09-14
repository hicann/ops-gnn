# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""按当前 NPU 芯片型号对应的构建架构，只收集对应测试目录。

    test/<operator>/arch22/  -> A2(910B)/A3(910C) 构建（dav-2201）
    test/<operator>/arch35/  -> 950 构建（dav-3510）
"""

import os


def _built_npu_arch() -> str:
    arch = os.getenv("NPU_ARCH") or os.getenv("TARGET_NPU_ARCH") or ""
    try:
        from ops_gnn import _pybind

        return getattr(_pybind, "npu_arch", arch)
    except ImportError:
        return arch


_IGNORED_ARCH = "arch35" if _built_npu_arch() == "dav-2201" else "arch22"


def pytest_ignore_collect(collection_path, config):
    """Ignore architecture subdirectories that do not match the built extension."""
    return _IGNORED_ARCH in collection_path.parts
