# CrossPascal Converter

## Native Windows converter

Open `CrossPascalConvert.exe` from the Windows x64 SDK. Select a profile,
provide the original checkpoint and choose a new output directory. For a
Gemma 3n image bundle, also provide the matching projector GGUF. Use
**Preview** to review the selected build, then **Build** to generate TVM
libraries and a model bundle. The desktop launcher and its converter tools
run without Python. The SDK includes the TVM compiler. Conversion uses the
Windows x64 linker and SDK libraries, with NVRTC for CUDA builds.

See the [installation guide](INSTALL.md) for the current profiles and toolchain. The sections below describe the browser converter from earlier releases.

The native converter includes **BiRefNet HR800 background removal | CPU** and **CUDA**. Choose the pinned full SwinL ONNX from the BiRefNet source pack and a new output folder. The resulting bundle works with the [Pascal BiRefNet example](https://github.com/pisarev/crosspascal/tree/main/examples/birefnet), which saves a grayscale mask or transparent PNG.

## Browser converter in earlier releases

CrossPascal Convert builds native model libraries and weights for Pascal applications. The converter runs on your computer and opens a local browser interface. This repository contains its Python package, browser interface, model adapters, and language-model compiler integration.

## Get started

Use the Windows x64 SDK from [CrossPascal Releases](https://github.com/pisarev/crosspascal/releases). It supplies the launcher, portable Python, and sibling Pascal sources. Follow the [installation guide](INSTALL.md) for prerequisites and the first Qwen and CPU vision examples in Delphi 13.2 or Lazarus 4.8.

1. Open `CrossPascalConvert.exe` from the extracted SDK.
2. Select the pinned Qwen source or an ONNX file from the matching CPU model source pack.
3. For Qwen, accept the detected CUDA architecture or enter the destination architecture. For a CPU pack, retain the profile's supplied settings.
4. Review the source, target, runtime, and output directory in Preview. Use a new output directory for the build.
5. Build the bundle and open it in the corresponding Pascal example.

Choose **Quit converter** when you finish. Closing the browser tab leaves the local service running. Downloads and default build outputs go to the user's local application-data directory.

## Scope

The named Qwen2.5-Instruct CUDA profiles support 0.5B (24 layers) and 1.5B (28 layers). Both use FP32, batch size one, and a 64-token context and prefill chunk. The wizard checks the selected official source revision, complete source-tree SHA-256 and profile-specific shape before Build. The same installed TVM runtime serves both profiles. The CUDA architecture is selected separately. Run QwenChat after conversion to check inference on the destination GPU. See [source notes](docs/SOURCE-NOTES.md).

### Experimental Gemma conversion

Enter a local Gemma 2 checkpoint directory or a Gemma 3n GGUF file in the model path field and open **Gemma conversion (experimental)**. Choose a context size and execution target, click **Inspect Gemma checkpoint**, then click **Convert Gemma**. For Gemma 3n, automatic selection detects CUDA when available; a manual destination architecture is also supported. The wizard derives the topology from the configuration, architecture metadata and tensor shapes. No model-name catalog entries are required. See [Gemma 3n setup and scope](docs/GEMMA3N.md).

The experimental Gemma 2 CPU route accepts compatible text checkpoints with the expected tokenizer and contexts up to 1,024 tokens. The preview reports when a Gemma architecture requires another backend. Build confirms conversion and reports the bundle directory. Run the Pascal consumer to check inference with the converted model.

Gemma 3n has a separate native TVM text backend with AltUp, LAuReL, per-layer embeddings, activation sparsity, alternating local/global attention, shared KV layers and output soft-capping. Its source format is GGUF. The CPU profile uses file-backed FP32 parameters. The CUDA profile uses compact int8 projection weights with FP32 arithmetic and host embedding lookup. GPU memory requirements depend on the architecture and compiled context; available memory is checked before loading. For CPU or CUDA text bundles with host embeddings, the experimental image add-on converts a matching Gemma 3n projector and creates an independent text-and-image bundle. Audio and arbitrary other Gemma architectures are outside this route. Compilation success alone does not establish model quality or performance.

An additional **compatible local Qwen2 dense-text** profile is available for explicitly selected local original Safetensors folders. It derives supported topology and resource requirements from verified configuration and tensor headers instead of a model-specific name or official-source catalog entry. The local route retains Windows x64, CUDA FP32, one batch, a 2,048-token context and the exact established Qwen byte-BPE tokenizer semantics. Use the **Compatible local Qwen2** dropdown entry, inspect the Preview memory/disk requirements and source SHA-256, and supply the existing native Qwen tokenizer/runtime kit; this kit does not establish an official identity for the selected local weights. It is **not** general support for Qwen-VL, MoE, arbitrary tokenizers, quantized formats or arbitrary context lengths. The Pascal consumer checks bundle structure and available GPU memory before loading.

The documented CPU routes cover SqueezeNet 1.1, ResNet50, YOLO11n classification, YOLO11n detection, and RT-DETR R18vd. Other families and compilation paths in the source tree are outside the verified early access scope.

A completed build confirms the converter's file checks. Run the Pascal example to confirm inference on your computer. For advanced use, see the [source notes](docs/SOURCE-NOTES.md).

## License

Original source and documentation use the [MIT license](LICENSE). Runtime dependencies and pretrained models retain their own terms.

## Qwen2.5-VL images on CPU

The separate Qwen2.5-VL route reads checkpoint configuration and tensor shapes
instead of matching model names. It converts the vision encoder, multimodal
positions, a 16-token prompt graph and cached token decoder to native TVM CPU
libraries. Completed bundles work through the same chat and console API as
the other supported language architectures. Dense Qwen2.5 remains text-only.
See [Qwen2.5-VL conversion](docs/QWEN2.5-VL.md) for manual steps, memory needs,
model terms and the distinction between executed checkpoints and configuration
validation. CPU image inference is an optional local route; it is not a claim
of GPU-like speed or support for every checkpoint.
