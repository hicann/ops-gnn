# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent PyTorch reference, usable on CPU without ops_gnn."""
import torch


def _align_operands(lhs, rhs):
    feature_rank = max(lhs.dim(), rhs.dim()) - 1
    lhs_padding = feature_rank - (lhs.dim() - 1)
    rhs_padding = feature_rank - (rhs.dim() - 1)
    lhs = lhs.reshape(
        (lhs.shape[0],) + (1,) * lhs_padding + tuple(lhs.shape[1:]))
    rhs = rhs.reshape(
        (rhs.shape[0],) + (1,) * rhs_padding + tuple(rhs.shape[1:]))
    return lhs, rhs


def _message_values(indices, x, op, rhs):
    if op == "copy_rhs":
        return x
    gathered = x[indices.long()]
    if op == "copy_lhs":
        return gathered
    gathered, rhs = _align_operands(gathered, rhs)
    if op == "add":
        return gathered + rhs
    if op == "sub":
        return gathered - rhs
    if op == "mul":
        return gathered * rhs
    if op == "div":
        return gathered / rhs
    raise ValueError(f"unsupported message op: {op}")


def _sum_or_mean(rows, degrees, values, shape, reduce):
    result = torch.zeros(shape, dtype=values.dtype, device=values.device)
    result.index_add_(0, rows, values)
    if reduce == "mean":
        divisor = degrees.clamp(min=1).to(values.dtype)
        divisor = divisor.reshape((shape[0],) + (1,) * (values.dim() - 1))
        result = result / divisor
    return result


def _min_or_max(rows, degrees, values, shape, reduce):
    initial = -float("inf") if reduce == "max" else float("inf")
    result = torch.full(shape, initial, dtype=values.dtype, device=values.device)
    index = rows.reshape((-1,) + (1,) * (values.dim() - 1)).expand_as(values)
    reduction = "amax" if reduce == "max" else "amin"
    result.scatter_reduce_(0, index, values, reduce=reduction, include_self=True)
    result[degrees == 0] = 0
    return result


def _reduce_values(indptr, values, reduce):
    row_count = indptr.numel() - 1
    degrees = (indptr[1:] - indptr[:-1]).long()
    rows = torch.repeat_interleave(
        torch.arange(row_count, device=values.device), degrees)
    compute = values.float() if values.dtype == torch.float16 else values
    shape = (row_count,) + tuple(compute.shape[1:])
    if reduce in ("sum", "mean"):
        return _sum_or_mean(rows, degrees, compute, shape, reduce)
    if reduce in ("max", "min"):
        return _min_or_max(rows, degrees, compute, shape, reduce)
    raise ValueError(f"unsupported reduction: {reduce}")


def unified_spmm_reference(indptr, indices, x, *args, **kwargs):
    names = ("reduce", "op", "rhs")
    if len(args) > len(names):
        raise TypeError("unified_spmm_reference accepts at most six positional arguments")
    options = {"reduce": "sum", "op": "copy_lhs", "rhs": None}
    for name, value in zip(names, args):
        if name in kwargs:
            raise TypeError(f"multiple values for argument '{name}'")
        options[name] = value
    unexpected = set(kwargs) - set(names)
    if unexpected:
        name = sorted(unexpected)[0]
        raise TypeError(f"unexpected keyword argument '{name}'")
    options.update(kwargs)
    values = _message_values(indices, x, options["op"], options["rhs"])
    result = _reduce_values(indptr, values, options["reduce"])
    return result.to(values.dtype)
