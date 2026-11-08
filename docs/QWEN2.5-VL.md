# Qwen2.5-VL: a picture and a question

This route converts compatible Qwen2.5-VL checkpoints into native TVM CPU
libraries. Architecture, layer counts, tensor shapes, attention heads and image
tokens come from the checkpoint configuration. No list of model names or
graphics-card names selects the implementation. Dense Qwen2.5 is a different,
text-only architecture. Qwen3-VL, video and audio are outside this route.

## What to download

Obtain an original compatible checkpoint under its original model terms. Keep
its `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`preprocessor_config.json` and all Safetensors files in one folder. For a
sharded checkpoint, include `model.safetensors.index.json`. Keep any supplied
generation configuration, chat template and license. An incomplete download
is rejected before conversion.

Model terms are separate from the CrossPascal source-code license. The tested
Qwen2.5-VL-3B-Instruct checkpoint uses the Qwen Research License; it must not
be described as Apache-2.0 merely because other Qwen checkpoints use that
license. Check the terms accompanying the checkpoint you choose.

## Convert by hand

1. Prepare the converter's pinned TVM, LLVM and Windows C++ linker environment
   as described in [INSTALL.md](../INSTALL.md).
2. Prepare the exporter Python environment described in
   [Gemma conversion](GEMMA3N.md#conversion-dependencies), with Transformers
   4.57.6, PyTorch 2.11, Safetensors, ONNX and Tokenizers 0.22.2. The environment
   is used only for conversion; inference does not require these compilers or
   PyTorch. Set `CROSSPASCAL_GEMMA_PYTHON` to its Python executable.
3. Open the local converter and expand **Qwen2.5-VL text and images on CPU (experimental)**.
4. Select the original checkpoint folder. Choose the context and image size.
   Start with a 2048-token context and a 224-pixel image profile.
5. Keep the LLVM CPU target **generic** for portable x64 code. A specific LLVM
   CPU target is optional and requires a compatible destination processor.
6. Inspect the checkpoint. Check its detected architecture, tensor dimensions
   and estimated storage and memory requirements.
7. Convert it. The converter verifies that the source is unchanged after
   inspection and creates a separate bundle. Do not rename individual bundle
   files or mix files from different conversions.
8. In QwenChat or GemmaChat, select the completed folder containing `model.json`,
   click **Load model**, attach a picture and ask a question. The minimal console
   examples accept the same bundle and an optional image filename.

## What the CPU profile does

The bundle contains separate native vision, prompt-prefill and token-decoding
graphs, file-backed FP32 weights and a local tokenizer. CPU inference uses TVM;
it does not call PyTorch, an online service or a hidden GPU fallback.

Image preprocessing preserves aspect ratio, uses bicubic interpolation and
pads the image to the selected square size. The source's normalization and
multimodal rotary positions are preserved. A 224-pixel profile uses 64 image
tokens. Supported image sizes must match the source patch/merge dimensions and
cannot exceed 448 pixels in this early profile. Larger inputs increase memory,
context consumption and work. A simple thumbnail question is a better first
test than reading small print in a large document.

New CPU bundles read prompt tokens in blocks of 16 and retain a KV cache for
decoding. Loaded weights survive repeated requests; an unchanged image can
reuse its vision features. Each request has a fresh conversational context.
The runtime reports loading, image encoding, first-token and total time.

FP32 expansion needs substantially more space than the original low-precision
checkpoint. The 3B checkpoint expands to approximately 15 GB of weights.
Available RAM, filesystem caching and storage speed strongly affect a cold
start. A CPU image route is suitable for occasional local tasks; it does not
promise GPU-like latency. The converter checks estimated memory and disk space
before beginning.

## Compatibility and limits

The 3B checkpoint has native image and text comparisons against PyTorch on the
same weights and preprocessing. A distinct 7B configuration passes topology
validation, but that is not an inference or performance measurement of 7B.
The consumer uses the standard Qwen chat markers. Arbitrary custom Jinja chat
templates are not executed; select plain completion explicitly for a base model.
Larger or custom compatible checkpoints still need enough resources and their
own execution checks. A successful build is recorded as `NOT_CHECKED` until
the resulting bundle is run.

Automatic output uses `0` new tokens and ends at EOS, Stop or the compiled
context boundary. A positive manual maximum has no artificial demo ceiling.
Text, image tokens and output all consume the same context. Unsupported
architectures, incomplete projectors, changed artifacts and text-only image
requests produce explicit errors.
