"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import pytest


def test_import():
    """测试基本导入"""
    import ops_gnn
    assert hasattr(ops_gnn, 'add_sample')
    assert hasattr(ops_gnn, 'graclus_cluster')


def test_version():
    """测试版本号"""
    import ops_gnn
    assert ops_gnn.__version__ == '0.1.0'