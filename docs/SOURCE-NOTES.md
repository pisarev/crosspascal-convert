# Converter source notes

This source snapshot contains the converter package, its browser interface, the existing family and support modules, and the native language-model builder. Packaging metadata and entry points are included, while the Windows SDK supplies portable Python and a pinned dependency installer. Inclusion of an adapter does not establish support for every model, toolchain, or machine it can describe.

## Named language-model route

The pinned Qwen2.5 CUDA profiles are `qwen2.5-0.5b-cuda-f32-64-v1` (24 layers, hidden 896, intermediate 4864, 14 attention heads, 2 KV heads, head width 64) and `qwen2.5-1.5b-cuda-f32-64-v1` (28 layers, hidden 1536, intermediate 8960, 12 attention heads, 2 KV heads, head width 128). Both use FP32, batch size 1, and context and prefill capacities of 64 tokens. BuildPlan records the resolved CUDA architecture separately from the model profile. The native builder checks the plan before compilation.

The official Qwen2.5-0.5B-Instruct source is pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775` and tree SHA-256 `90361a89884fd025cac6718a75d8ede78d2e259d2be5bef2e33d78f08f8585b1`. The official Qwen2.5-1.5B-Instruct source is pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and tree SHA-256 `5f0180b3ad3b8652860444471fe9e2dd2e5ce325e5e5dccd2be7190ef437cf0d`. The latter includes `model.safetensors` (3,087,467,144 bytes; SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`). Selection and download check every required source size and SHA-256. A familiar directory name or an otherwise matching configuration alone is never proof of an official source. The builder records model source identity and computes the actual serialized weight digest during conversion.

The two official revisions have byte-identical upstream `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt`. Their source-tree digests are **not** interchangeable: configuration and model weights differ. A Qwen 1.5B consumer kit must provide its own pinned license and `MODEL-SOURCE.json`; previously exported tokenizer binaries may be used only when the held native input and output bytes pass the existing selected-HF tokenizer binding. Reusing a filename or model name is not adequate. Both use the existing `qwen2-cuda.dll` ABI, the same five tokenizer output filenames, and the fixed embedding to seven packed tensors per layer to final normalization order. New `model.json` files include `intermediate_size`; the published 0.5B format without that field is still understood.

The named route requires the selected consumer kit and its tokenizer and license files before compilation. It retains the distinction between build success, bundle consistency, source identity, target identity, and consumer execution. Missing required checks remain `NOT_RUN`; diagnostic results cannot turn a required failure into a pass. The build receipt leaves consumer execution `NOT_RUN`. Existing first-run validation evidence pertains only to the 0.5B profile; the new 1.5B converter route has not been validated by Delphi/Lazarus inference in this source task. This aggregation applies to the named LLM route and does not redesign the generic or multi-input verifier.

### Source-neutral compatible Qwen2 text candidate

The additional `qwen2-compatible-local` route has one versioned producer/consumer contract: `model.json.qwen_compatibility.schema = crosspascal.qwen2-compatible.v1`. The associated BuildPlan profile is `qwen2-compatible-cuda-f32-2048-v1`, not a checkpoint-specific profile ID. The input must be a local, original, single or indexed-shard Safetensors tree containing a dense `Qwen2ForCausalLM` text model with SiLU/SwiGLU, regular RoPE, RMSNorm, a supported GQA arrangement and the exact known Qwen byte-BPE `tokenizer.json` content. The accepted tokenizer SHA-256 is `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`. The converter never runs the source's custom code, remote templates, Python modules, or remote downloads.

The local contract supports either tied embeddings (embedding plus seven packed parameters per layer plus final norm) or an independent untied `lm_head.weight` (one additional final parameter). It preserves the existing four packed Qwen2 runtime entry points and native file names. The canonical description includes hidden/intermediate sizes, layers, query/KV head counts, RoPE theta, RMSNorm epsilon, a full measured `source_tree_sha256`, the operator and tokenizer contract IDs, exact packed byte count, and lower-bound FP32 disk/host RAM/GPU VRAM requirements. The Python builder checks Safetensors tensor names/dtypes/shapes, shard closure and full streaming SHA-256 under Win32-held private source custody. QwenChat independently checks the descriptor and Qwen tensor/index layout, hashes the binary outputs and checks target CUDA architecture and free memory before execution. Source-tree identity is local byte custody, **not** an upstream release/repository attestation or a signed publisher identity. The bundle records local weights in `model.json` and `import-provenance.json`. The EA009 tokenizer and runtime kit supplies its own `TOKENIZER-KIT-SOURCE.json` and notices. Keep the license for the selected checkpoint separately.

Limits: Windows x64 CUDA FP32, batch one, 2,048-token context/prefill, vocabulary 151936, known BOS/EOS, 1-80 layers, bounded 128-8192 hidden width and 128-32768 intermediate width, even RoPE head width 32-256, no more than 12 GiB final packed FP32 parameters. Source weights may be F16, BF16 or F32 but mixed tensors, unsupported quantized formats, altered tokenizer files, unusual RoPE variants, MoE, sliding-window attention, and multimodal inputs are refused. The memory estimates are minimum admission limits, not predictions of full TVM compilation peak or universal GPU support. A successful Python smoke test does not constitute Delphi/Lazarus or CUDA acceptance.

Captured Python child processes use the selected console interpreter. When the interface runs through `pythonw.exe`, the sibling `python.exe` is required rather than an unrelated interpreter from PATH. The generic verifier uses the same interpreter selection for its child processes.

## Inputs and model contracts

Input names, shapes, and types must come from the selected model. A shape hint uses the last colon as its separator because tensor names may already contain a colon, such as an output suffix from TensorFlow. Unknown input names and unresolved dimensions should be reported explicitly.

File contents matter as well as filenames: the TFLite identifier follows its root-table offset, and a downloaded model may have a generic filename. A directory-based model, an ONNX graph with external tensor files, and a self-contained graph have different dependency requirements. The selected-closure helpers cover only their documented file sets and size limits.

A build recipe records a request. A manifest describes the consumer-facing model contract. Compiler success, valid metadata, and a completed numerical comparison are separate facts. A multi-input model must not receive a single-input manifest, and an adapter's declared output shape must also match its decoding semantics.

## Targets and tuning

The deployment OS, CPU architecture, GPU architecture, and driver requirements are distinct from the build host. Unknown or unsupported combinations must be refused or left for a suitable runner. Compilation on one machine does not prove execution on another.

The selected CPU tuning route uses a local runner, so a compiler target that the local processor cannot execute is not a valid tuning choice. A tuning database must cover the actual model's tasks with measured records; file presence and unmeasured sentinel records do not establish coverage. Historical timings in the source describe specific observations, not a general performance promise.

The cuDNN helper still lacks a complete guard for grouped convolution.

## File handling and runtime bundles

A checked copy is not an immutable snapshot. Source-freeze and selected-file helpers document the limits of their scans, including changes between observations. Win32 holders keep handles for a defined scope and perform their post-audit before releasing them; their receipts do not establish model accuracy or release-grade immutability.

Private stages and final output names have separate roles. Existing output files must not be overwritten, and a directory name alone does not authorize cleanup of its contents. An atomic namespace commit does not promise durability after power loss.

Runtime inventory ties selected library bytes to the building environment and includes required license files. Untracked libraries in the destination can change what loads. An exact inventory does not prove compatibility with another device, driver, or operating system.

## Comparisons and diagnostics

The included verifier has several comparison profiles with different rules. Output positions, candidate-row associations, and class vectors must retain their documented meaning; a successful comparison under one profile must not be presented as success under another. Synthetic inputs are not a substitute for representative images or task-specific acceptance.

In the generic probe path, matching labels or confidence-threshold decisions are diagnostic observations. They cannot cancel a failed numeric comparison of scores, boxes, or another output. Judging probes also reject invalid shapes, types, empty tensors, and nonfinite values. A deliberately nonjudging noise probe retains its diagnostic role; an output-count mismatch remains a required failure for every probe.

The generic and multi-input profiles still need explicit selection from a complete consumer contract. Some selection and row-matching rules remain heuristic. Their outer reporting also needs consistent separation of an execution error, an unattempted comparison, and a numeric mismatch. These remain tasks for those routes even though the new named LLM route has an explicit profile and required-check aggregation. OmDet remains outside the supported Early Access conversion routes.

A comparison that did not run does not establish acceptance. Native library loading can outlive Python objects, which is why the source isolates certain checks in a child process and checks its actual exit. Do not invoke these paths as an import smoke test.

The sample measurements are dated data retained from the source snapshot. Run pascalvision-convert support-inventory for the documented Early Access routes. This command reports the distribution scope and does not test the current machine.
