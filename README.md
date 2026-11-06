# CrossPascal Converter

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

The current native backend builds CPU FP32 bundles for compatible Gemma 2 text checkpoints with the expected tokenizer and contexts up to 1,024 tokens. This experimental short-context implementation omits attention-score soft-capping. The preview reports when a Gemma architecture requires another backend. Build confirms conversion and reports the bundle directory. Inference requires a separate check.

Gemma 3n has a separate native TVM text backend with AltUp, LAuReL, per-layer embeddings, activation sparsity, alternating local/global attention, shared KV layers and output soft-capping. Its source format is GGUF. The CPU profile uses file-backed FP32 parameters. The CUDA profile uses compact int8 projection weights with FP32 arithmetic and host embedding lookup. GPU memory requirements depend on the architecture and compiled context; available memory is checked before loading. Image/audio execution and arbitrary other Gemma architectures are outside this route. Compilation success alone does not establish model quality or performance.

An additional **compatible local Qwen2 dense-text** profile is available for explicitly selected local original Safetensors folders. It derives supported topology and resource requirements from verified configuration and tensor headers instead of a model-specific name or official-source catalog entry. The local route retains Windows x64, CUDA FP32, one batch, a 2,048-token context and the exact established Qwen byte-BPE tokenizer semantics. Use the **Compatible local Qwen2** dropdown entry, inspect the Preview memory/disk requirements and source SHA-256, and supply the existing native Qwen tokenizer/runtime kit; this kit does not establish an official identity for the selected local weights. It is **not** general support for Qwen-VL, MoE, arbitrary tokenizers, quantized formats or arbitrary context lengths. The Pascal consumer checks bundle structure and available GPU memory before loading.

The documented CPU routes cover SqueezeNet 1.1, ResNet50, YOLO11n classification, YOLO11n detection, and RT-DETR R18vd. Other families and compilation paths in the source tree are outside the verified early access scope.

A completed build confirms the converter's file checks. Run the Pascal example to confirm inference on your computer. For advanced use, see the [source notes](docs/SOURCE-NOTES.md).

## License

Original source and documentation use the [MIT license](LICENSE). Runtime dependencies and pretrained models retain their own terms.
