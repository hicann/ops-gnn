"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import os
import pytest
import torch
import ops_gnn


def test_add_sample_npu():
    """测试add_sample NPU算子"""
    device_id = int(os.environ.get("NPU_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch.manual_seed(42)
    shape = (1024, 1024)
    src1 = torch.randint(0, 128, shape, dtype=torch.uint8).npu()
    src2 = torch.randint(0, 128, shape, dtype=torch.uint8).npu()
    
    result = ops_gnn.add_sample(src1, src2)

    expected = src1.cpu() + src2.cpu()
    assert result.device.type == 'npu'
    assert result.shape == shape
    assert torch.equal(result.cpu(), expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
