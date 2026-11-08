# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Schedule FP32 row-major dot products without gathering across rows."""

from tvm import s_tir, tirx
from tvm.s_tir import dlight


def schedule_gemma_cpu(mod):
    for symbol, func in list(mod.functions.items()):
        if not isinstance(func, tirx.PrimFunc):
            continue
        if func.attrs.get("tirx.is_scheduled", False):
            continue
        if not (symbol.name_hint.startswith("NT_matmul") or
                symbol.name_hint.startswith("fused_NT_matmul")):
            continue
        buffers = [param for param in func.params if isinstance(param, tirx.Buffer)]
        if len(buffers) != 3:
            continue
        x, weight, y = buffers
        if any(str(buffer.dtype) != "float32" for buffer in buffers):
            continue
        if len(x.shape) != 3 or len(weight.shape) != 2 or len(y.shape) != 3:
            continue
        fixed = (x.shape[0], x.shape[-1], *weight.shape, y.shape[0], y.shape[-1])
        if not all(isinstance(dim, tirx.IntImm) for dim in fixed):
            continue
        if not x.shape[1].same_as(y.shape[1]):
            if not (isinstance(x.shape[1], tirx.IntImm) and isinstance(y.shape[1], tirx.IntImm)
                    and int(x.shape[1]) == int(y.shape[1])):
                continue
        if int(x.shape[0]) != 1 or int(y.shape[0]) != 1:
            continue
        if int(x.shape[-1]) != int(weight.shape[-1]) or int(y.shape[-1]) != int(weight.shape[0]):
            continue
        if int(weight.shape[-1]) % 8:
            continue

        schedule = s_tir.Schedule(func)
        block = schedule.get_sblock("NT_matmul", func_name="main")
        _, _, _, reduction = schedule.get_loops(block)
        _, lanes = schedule.split(reduction, factors=[None, 8])
        partial = schedule.rfactor(lanes, factor_axis=3)
        b, seq, rows, chunks, lane = schedule.get_loops(partial)
        schedule.reorder(b, seq, rows, chunks, lane)
        schedule.parallel(schedule.fuse(seq, rows))
        schedule.vectorize(lane)
        finished = schedule.get_loops(block)
        schedule.parallel(schedule.fuse(finished[1], finished[2]))
        schedule.unroll(finished[-1])
        mod.update_func(symbol, schedule.mod["main"].with_attr("tirx.is_scheduled", True))
        print(f"Row-major SIMD scheduled: {symbol.name_hint}", flush=True)
    return dlight.ApplyDefaultSchedule(dlight.cpu.GEMV(), dlight.cpu.Reduction())(mod)
