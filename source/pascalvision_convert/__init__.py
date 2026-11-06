# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
"""pascalvision-convert: ONNX -> TVM Relax converter for crosspascal-vision."""

__version__ = "0.2.0a1"

from .convert import Target, convert_onnx

__all__ = ["Target", "convert_onnx", "__version__"]
