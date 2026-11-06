# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""Keep real-valued x**2 valid for negative x under CUDA fast math."""
import numpy as np
from onnx import numpy_helper


def replace_squares(graph):
    constants = {v.name: numpy_helper.to_array(v) for v in graph.graph.initializer}
    for node in graph.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    count = 0
    for node in graph.graph.node:
        if node.op_type == 'Pow' and node.input[1] in constants:
            exponent = np.asarray(constants[node.input[1]])
            if exponent.size == 1 and exponent.item() == 2:
                node.op_type = 'Mul'
                node.input[1] = node.input[0]
                count += 1
    return count
