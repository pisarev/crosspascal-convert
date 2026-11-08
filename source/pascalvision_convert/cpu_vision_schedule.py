# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""CPU vision schedules for NCHW convolutions, normalization and dot products."""
from tvm import s_tir, tirx


def _schedule_channel_norm(func):
    """Keep channel reductions ordered, but read adjacent image pixels together."""
    schedule = s_tir.Schedule(func)
    root = schedule.get_sblock("root", func_name="main")
    blocks = list(schedule.get_child_blocks(root))
    reductions = []
    for block in blocks:
        node = schedule.get(block)
        kinds = [int(v.iter_type) for v in node.iter_vars]
        if 2 in kinds:
            if kinds != [0, 0, 0, 0, 2] or len(node.reads) != 1:
                return None
            source, result = node.reads[0].buffer, node.writes[0].buffer
            if len(source.shape) != 4 or len(result.shape) != 4:
                return None
            if not isinstance(result.shape[1], tirx.IntImm) or int(result.shape[1]) != 1:
                return None
            reductions.append(block)
    if len(reductions) != 1:
        return None
                                                                                 
                                                                            
    for block in reversed(blocks[:-1]):
        if all(int(v.iter_type) == 0 for v in schedule.get(block).iter_vars):
            schedule.compute_inline(block)
    for block in schedule.get_child_blocks(root):
        node = schedule.get(block)
        loops = list(schedule.get_loops(block))
        if any(int(v.iter_type) == 2 for v in node.iter_vars):
            batch, channel, height, width, reduction = loops
            outer, lane = schedule.split(width, factors=[None, 8])
            schedule.reorder(batch, channel, height, outer, reduction, lane)
            schedule.parallel(schedule.fuse(batch, channel, height))
            schedule.vectorize(lane)
        else:
            fused = schedule.fuse(*loops)
            outer, lane = schedule.split(fused, factors=[None, 8])
            schedule.parallel(outer)
            schedule.vectorize(lane)
    return schedule.mod["main"].with_attr("tirx.is_scheduled", True)


def schedule_cpu_vision(module):
    for symbol, func in list(module.functions.items()):
        if not isinstance(func, tirx.PrimFunc) or func.attrs.get("tirx.is_scheduled", False):
            continue
        if symbol.name_hint.startswith("fused_mean"):
            normalized = _schedule_channel_norm(func)
            if normalized is not None:
                module.update_func(symbol, normalized)
                print("CPU channel normalization scheduled: " + symbol.name_hint, flush=True)
                continue
        schedule = s_tir.Schedule(func)
        root = schedule.get_sblock("root", func_name="main")
        changed = False
        for block in schedule.get_child_blocks(root):
            node = schedule.get(block)
            name = node.name_hint
            loops = list(schedule.get_loops(block))
            kinds = [int(v.iter_type) for v in node.iter_vars]
            if name in ("conv2d_nchw", "group_conv2d_nchw") and len(loops) == 7 and kinds == [0, 0, 0, 0, 2, 2, 2]:
                n, channel, height, width, rc, ry, rx = loops
                width_outer, lane = schedule.split(width, factors=[None, 8])
                schedule.reorder(n, channel, height, width_outer, rc, ry, rx, lane)
                schedule.parallel(schedule.fuse(n, channel, height))
                schedule.vectorize(lane)
                changed = True
            elif name == "NT_matmul" and kinds[-1:] == [2] and all(v == 0 for v in kinds[:-1]):
                extent = schedule.get(loops[-1]).extent
                if not isinstance(extent, tirx.IntImm) or int(extent) % 8:
                    continue
                _, lane = schedule.split(loops[-1], factors=[None, 8])
                partial = schedule.rfactor(lane, factor_axis=len(kinds) - 1)
                partial_loops = list(schedule.get_loops(partial))
                if len(partial_loops) >= 5:
                    prefix, sequence, row, reduction, vector = partial_loops[:-4], *partial_loops[-4:]
                    schedule.reorder(*prefix, row, sequence, reduction, vector)
                    schedule.parallel(schedule.fuse(*prefix, row))
                else:
                    schedule.parallel(schedule.fuse(*partial_loops[:-2]))
                schedule.vectorize(partial_loops[-1])
                final_loops = list(schedule.get_loops(block))
                schedule.parallel(schedule.fuse(*final_loops[:-1]))
                schedule.unroll(final_loops[-1])
                changed = True
        if changed:
            module.update_func(symbol, schedule.mod["main"].with_attr("tirx.is_scheduled", True))
            print("CPU vision scheduled: " + symbol.name_hint, flush=True)
    return module
