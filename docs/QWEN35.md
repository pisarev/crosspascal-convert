# Qwen3.8-27B on Windows

CrossPascal runs the dense Qwen3.8 text decoder and, in image bundles, the
original vision encoder through native TVM graphs.
Each bundle records its compiled context in `model.json`. That budget includes
the chat template, image tokens, your prompt, and the generated answer.

| Source | Ready modes | Pinned source revision |
| --- | --- | --- |
| `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_M.gguf` | CPU and CUDA | `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` |
| `Qwen/Qwen3.8-27B`, original BF16 shards | CUDA with weight streaming | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |

The original GGUF contains a mixture of nine tensor formats. Its bytes stay
unchanged. The BF16 profile keeps the original 18 shard files and their
weight precision. TVM computes with FP32 accumulators and streams the
weights needed for each layer. Model files retain their Apache-2.0 terms
and the supplied dependency notices.

## Prepare and run

1. Download a dedicated Qwen3.8 Q4 CPU, Q4 GPU, or BF16 GPU Prepare kit from
   the [EA023 release](https://github.com/pisarev/crosspascal/releases/tag/early-access-023).
2. Extract the ZIP to a writable folder and run **Prepare.cmd**. The dedicated
   kit selects its model and device. It downloads the SDK, native artifacts,
   and original source weights, checking every file's size and SHA-256.
3. Open the printed `QwenChat.dproj` in Delphi or `QwenChat.lpi` in Lazarus.
   Build for Windows 64-bit and run the project. You can also run the supplied
   Lazarus executable from its prepared project folder.
4. Click **Load model**, enter your message, optionally attach an image, and
   click **Generate**.
   **Stop** cancels generation; **Unload** releases the model.

The ready bundles accept text and images and have a 4,096-token context.
The CUDA packages contain code for `sm_75`, `sm_86`, `sm_89`, and `sm_120`.
Prepare reads the installed device's compute capability and free memory.
Keep the prepared SDK and model folders together. Downloads resume through
the shared cache when preparation is repeated.

## Convert another local checkpoint

Open **CrossPascalConvert.exe** in the Windows SDK. Choose one of:

* **Qwen 3.5/3.8 dense GGUF | CPU**
* **Qwen 3.5/3.8 dense GGUF | CUDA**
* **Qwen 3.5/3.8 original BF16 | CUDA**

Select the original GGUF file or BF16 checkpoint folder. Supply a source
identity JSON beside the corresponding model license. For example:

```json
{
  "source_repository": "Qwen/Qwen3.8-27B",
  "source_revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
  "license_file": "LICENSE"
}
```

For the GGUF source in the table, use its Unsloth repository and revision.
A relative license path resolves beside the identity JSON. Choose a new
output folder outside the source checkpoint directory. For CUDA, the target
field starts with the architecture read from the installed NVIDIA driver;
you can select a different target for cross-compilation.

### GUI source identity with images

To build a bundle that also accepts images, add these seven optional fields
to the source identity JSON:

```json
{
  "vision_source": "VISUAL_SHARD.safetensors",
  "vision_config": "config.json",
  "vision_source_repository": "ORIGINAL_ORGANIZATION/ORIGINAL_MODEL",
  "vision_source_revision": "PINNED_40_OR_64_HEX_REVISION",
  "vision_license_file": "LICENSE",
  "vision_height": 512,
  "vision_width": 512
}
```

Relative `vision_source`, `vision_config`, and `vision_license_file` paths
resolve beside the identity JSON, like `license_file`. The fields have the
same meaning and checks as in [Optional image input](#optional-image-input);
other `vision_*` names are rejected. Without these fields the converter
builds a text-only bundle. **Preview** prints `Inputs: text`, or the image
canvas and the number of original BF16 vision tensors that match the vision
graph.

Click **Preview** to inspect the source dimensions, tensor bindings, and
attention layout. Click **Build** to copy and hash the original files,
prepare the tokenizer, and compile the native graphs. The output folder
appears when conversion succeeds. CUDA conversion uses the installed CUDA
Toolkit and the detected Visual Studio and Windows SDK tools.

The converter reads topology and tensor formats from the checkpoint. Its
dense Qwen adapter covers Gated DeltaNet, gated full attention, partial
RoPE, convolution, normalization, MLP, and output projection. The sources
in the table have 64 text layers: 48 linear-attention layers and 16 full
attention layers. Generated bundles record the actual shapes, source
identity, and artifact hashes.

## Native JSON requests

The native tool is `runtime/converter/ConvertQwen35.exe` in the SDK.
Use `ConvertQwen35.exe --inspect request.json` to inspect a source and
`ConvertQwen35.exe request.json` to convert it. Save requests as UTF-8.

A GGUF CUDA request can enable decoded-buffer reuse with
`cuda_decoded_buffer_pool: true`. This option reuses GPU buffers while
keeping the original packed weights. Its default is `false`.
`cuda_packed_cache_bytes` controls the separate packed-weight cache.
Generation reports expose that cache's budget, resident bytes, hits,
misses, and pressure clears. The counters cover the loaded model's lifetime.

The full request example below uses explicit installation paths. Replace
them with paths from your SDK, source checkpoint, Visual Studio, Windows
SDK, and CUDA installation. Set `DETECTED_CUDA_SM` to the target compute
capability, such as the value shown by the desktop conversion master.

```json
{
  "source": "C:/Models/Qwen3.8-27B-UD-Q4_K_M.gguf",
  "destination": "C:/Converted/qwen-q4-cuda",
  "context": 4096,
  "head_rows": 4096,
  "target": "cuda",
  "cuda_arch": "DETECTED_CUDA_SM",
  "source_repository": "unsloth/Qwen3.8-27B-GGUF",
  "source_revision": "4af51b3542f775b0a8e4b310bfb3f17e01c306c9",
  "license_file": "C:/Models/LICENSE-Qwen.txt",
  "ggml": "C:/CrossPascal/runtime/converter/ggml-base.dll",
  "packed_kernel_directory": "C:/CrossPascal/runtime/converter/kernels/qwen35-packed",
  "reader_licenses": {
    "LICENSE-GGML.txt": "C:/CrossPascal/runtime/converter/licenses/qwen35/LICENSE-GGML.txt",
    "LICENSE-GGML-vendors.txt": "C:/CrossPascal/runtime/converter/licenses/qwen35/LICENSE-GGML-vendors.txt",
    "LICENSE-LLVM.txt": "C:/CrossPascal/runtime/converter/licenses/qwen35/LICENSE-LLVM.txt",
    "LICENSE-OLLAMA.txt": "C:/CrossPascal/runtime/converter/licenses/qwen35/LICENSE-OLLAMA.txt",
    "SOURCE-GGML-reader.txt": "C:/CrossPascal/runtime/converter/licenses/qwen35/SOURCE-GGML-reader.txt"
  },
  "kit": {
    "compiler": "C:/CrossPascal/runtime/converter/compiler",
    "linker": "C:/BuildTools/bin/Hostx64/x64/link.exe",
    "ffi_library": "C:/CrossPascal/runtime/converter/compiler/tvm_ffi.lib",
    "msvc_libraries": "C:/BuildTools/lib/x64",
    "ucrt_libraries": "C:/WindowsSDK/Lib/ucrt/x64",
    "windows_libraries": "C:/WindowsSDK/Lib/um/x64",
    "nvrtc_library": "C:/CUDA/bin/nvrtc64_120_0.dll"
  },
  "cuda_packed_cache_bytes": 2147483648,
  "cuda_decoded_buffer_pool": true
}
```

Save the adjusted file as `request.json` and run the native commands above.
The desktop master uses its detected toolchain and the default buffer-pool
setting. The JSON route exposes the additional cache and buffer controls.

## Optional image input

A native JSON request can add an image encoder to a GGUF CPU, GGUF CUDA, or
original BF16 CUDA conversion. The text source stays exactly as in the
text-only request: the original Q4 GGUF or BF16 text weights are preserved,
and the text engine, context, and cache settings do not change. The image
encoder is built from the original BF16 `model.visual.*` tensors of the
original model repository. Their bytes are copied unchanged into
`vision-parameters.bin`; nothing is requantized or converted on disk.

Add these fields to the request. Replace the placeholders with your local
paths and the pinned identity of the repository that holds the visual
tensors:

```json
{
  "vision_source": "C:/Models/QwenOriginal/VISUAL_SHARD.safetensors",
  "vision_config": "C:/Models/QwenOriginal/config.json",
  "vision_source_repository": "ORIGINAL_ORGANIZATION/ORIGINAL_MODEL",
  "vision_source_revision": "PINNED_40_OR_64_HEX_REVISION",
  "vision_license_file": "C:/Models/QwenOriginal/LICENSE",
  "vision_height": 512,
  "vision_width": 512
}
```

* `vision_source` is the single original safetensors shard that holds every
  `model.visual.*` tensor. The shard index of the original repository names it.
* `vision_config` is the original, complete model `config.json` with both
  `vision_config` and `text_config`. Its `text_config.hidden_size` must equal
  the hidden size of the text source.
* `vision_source_repository`, `vision_source_revision`, and
  `vision_license_file` describe the source of the visual weights. They can
  differ from the text source identity, for example when the text comes from
  a GGUF repository. The revision is a pinned 40- or 64-character
  hexadecimal commit.
* `vision_height` and `vision_width` are optional and default to 512. Both
  must be positive multiples of `patch_size` × `spatial_merge_size` from the
  config.

Without `vision_source` the request stays text-only. The converter then
rejects any other `vision_*` field with an error instead of ignoring it.
Requests without vision fields behave and produce bundles as before.

`--inspect` checks the vision request without compiling or writing files. It
builds the static vision graph, matches every graph weight with a BF16 source
tensor of the same name and shape, and reports `vision_requested`,
`vision_supported`, and a `vision_encoder` summary with the image size,
tensor count, and weight bytes. A text-only request reports
`vision_supported: false`.

Conversion always copies the visual weights, also when `hardlink_weights` is
set, and requires free space for them plus compiler working space. CPU
bundles compile a CPU vision library. CUDA bundles use the same `cuda_arch`
and NVRTC library as the text graphs. The completed bundle adds:

| File | Content |
| --- | --- |
| `vision.dll` | TVM library of the image encoder |
| `vision-parameters.bin` | Unchanged BF16 bytes of every visual tensor, in graph input order |
| `vision.json` | Vision manifest: image size, tensor layout, file hashes, and source shard SHA-256 |
| `vision-config.json` | Copy of the supplied `vision_config` |
| `LICENSE-vision-model.txt` | Copy of the supplied vision license |

`model.json` then lists `modalities: ["text", "image"]`,
`vision_supported: true`, `vision_manifest: "vision.json"`, and
`vision_source` with the repository, revision, the shard SHA-256 measured
during conversion, and the license file. All five files are hashed in
`artifact_sha256`. The text source files, tokenizer, and text graphs are the
same as in a text-only conversion.

### Image preprocessing

The bundle uses one static CrossPascal profile. Each image is resized with
bicubic interpolation to fit the configured size while keeping its aspect
ratio. The image is centered and padded with RGB 0. Each RGB channel is
normalized with mean 0.5 and standard deviation 0.5. One image is encoded
at a time on the fixed grid, 512 × 512 pixels by default.

This profile is not the default Hugging Face `smart_resize` processor, and
its embeddings are not claimed to equal that processor's results.

The desktop examples accept images through the shared runtime when the
loaded bundle declares the image modality. Text-only bundles keep working
as before. Original BF16 text still requires CUDA; there is no default CPU
BF16 profile.
