# Converter sources and model contracts

The Windows x64 SDK provides a native desktop converter and six Pascal command-line converter tools. They compile model graphs into TVM libraries for Pascal applications. The [installation guide](../INSTALL.md) lists source formats, profiles, and the Windows compiler tools.

## Native OmDet-Turbo and vision profiles

The OmDet-Turbo CPU and CUDA profiles read the supplied source folder containing `text.onnx`, `detector.onnx`, `class-projection.f32`, `vocab.json`, and `merges.txt`. The source is the trained [omlab/omdet-turbo-swin-tiny-hf checkpoint](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf/tree/7fe93cecfb770c4d76cf71163956221249cab566), revision `7fe93cecfb770c4d76cf71163956221249cab566`.

The converter compiles the CLIP text encoder and Swin-Tiny detector into one bundle. Its manifest records the five runtime artifacts, their sizes and SHA-256 values, the CPU or CUDA target, and source provenance. The Pascal example accepts up to 16 editable class names and a task instruction. Text uses up to 77 CLIP tokens, including start and end tokens; the detector uses a 640-pixel RGB image input. See the [OmDet-Turbo example](https://github.com/pisarev/crosspascal/tree/main/examples/omdet-turbo).

BiRefNet uses the pinned full SwinL HR800 ONNX source and returns an original-size mask and transparent cutout. PP-OCRv5 Server has separate detector and recognizer profiles. The native vision profiles also include SqueezeNet 1.1, ResNet50, YOLO11n classification and detection, and RT-DETR R18vd. Each source pack includes its model identity and notices.

## Targets and runtime

The ready CUDA bundles cover `sm_75`, `sm_86`, `sm_89`, and `sm_120`, spanning GTX 1650 and RTX 20 through RTX 50 families. The desktop converter reads the installed device for the initial target. You can edit it to compile for another computer. Prepare selects a bundle by compute capability and checks the selected model's free-memory requirement.

Native conversion uses the bundled TVM compiler, an installed x64 MSVC linker, and Windows SDK libraries. CUDA compilation uses NVRTC from CUDA Toolkit 12.8. The native Pascal examples use the matching TVM runtime; prepared CUDA inference uses the installed NVIDIA driver.

The source recipes, bundle manifests, and runtime inventories record their respective inputs and outputs. A bundle manifest describes input names, types, shapes, artifact hashes, and the selected device target. Keep each bundle and its included notices together, and use a new destination folder when converting.

## Retained named Qwen profiles

The browser-converter sources retain the named Qwen profiles and their source records below. The current native desktop profiles are listed in the installation guide.

The pinned Qwen2.5 CUDA profiles are `qwen2.5-0.5b-cuda-f32-64-v1` (24 layers, hidden 896, intermediate 4864, 14 attention heads, 2 KV heads, head width 64) and `qwen2.5-1.5b-cuda-f32-64-v1` (28 layers, hidden 1536, intermediate 8960, 12 attention heads, 2 KV heads, head width 128). Both use FP32, batch size 1, and context and prefill capacities of 64 tokens. BuildPlan records the resolved CUDA architecture separately from the model profile. The native builder checks the plan before compilation.

The official Qwen2.5-0.5B-Instruct source is pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775` and tree SHA-256 `90361a89884fd025cac6718a75d8ede78d2e259d2be5bef2e33d78f08f8585b1`. The official Qwen2.5-1.5B-Instruct source is pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and tree SHA-256 `5f0180b3ad3b8652860444471fe9e2dd2e5ce325e5e5dccd2be7190ef437cf0d`. The latter includes `model.safetensors` (3,087,467,144 bytes; SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`). Selection and download check every required source size and SHA-256. A familiar directory name or an otherwise matching configuration alone is never proof of an official source. The builder records model source identity and computes the actual serialized weight digest during conversion.

Both official source records include the matching tokenizer, configuration, license, and model files. The two checkpoints share tokenizer bytes and use their own configuration and weight digests. The local dense-Qwen route records its selected source-tree SHA-256, topology, tokenizer contract, and CUDA target in the generated bundle.

## Licenses

Original converter code and documentation use the repository's MIT license. Model files, CLIP tokenizer data, TVM, and the other native dependencies retain their included upstream licenses and notices.
