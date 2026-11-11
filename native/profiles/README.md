# Conversion profiles

These files describe graph topology, tensor bindings and fixed input shapes.
The Gemma 3n and Qwen2.5-VL graphs take weights as inputs. The YOLO and
RT-DETR recipes have empty placeholders for trained tensors. The converter
reads the original checkpoint, binds the weights, and compiles the graph with
TVM. It needs the selected recipe at conversion time.

| Directory | Original checkpoint and profile |
| --- | --- |
| gemma3n | Gemma 3n E4B GGUF and its matching projector GGUF. The recipe describes the 768-pixel MobileNetV5 encoder. See the Gemma Terms and the timm license in crosspascal-convert/licenses. |
| qwen-vl | Qwen2.5-VL-3B-Instruct at revision 66285546d2b821cf421d4f5eb2576359d3770cd3, 224-pixel input, 2,048-token context. The Qwen Research License applies to the checkpoint. |
| yolo11n, yolo11n-cls | Ultralytics YOLO11n detection and classification, batch one, AGPL-3.0. The source-pack records identify the original .pt files and related source. |
| rtdetr-r18vd | PekingU/rtdetr_r18vd at revision ac77a11ff0170a41b771c03264987f8ce2b0d753, batch one, 640-pixel input, Apache-2.0. |
