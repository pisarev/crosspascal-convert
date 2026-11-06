# Converter source notes

This source snapshot contains the converter package, its browser interface, the existing family and support modules, and the native language-model builder. Packaging metadata and entry points are included, while the Windows SDK supplies portable Python and a pinned dependency installer. Inclusion of an adapter does not establish support for every model, toolchain, or machine it can describe.

## Named language-model route

The Qwen CUDA profile defines FP32, all 24 Qwen layers, batch size 1 and context and prefill capacities of 64 tokens. BuildPlan records the resolved CUDA architecture separately. The native builder checks the plan before compilation.

The official Qwen source is pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. Selection and download check its recorded source closure. A local model with a familiar directory name is not sufficient. The builder records model source identity and computes the serialized weight digest during the existing weight write; an older CPU or GPU weight digest is not substituted for that measurement.

The named route requires the selected consumer kit and its tokenizer and license files before compilation. It retains the distinction between build success, bundle consistency, source identity, target identity, and consumer execution. Missing required checks remain `NOT_RUN`; diagnostic results cannot turn a required failure into a pass. The build receipt leaves consumer execution `NOT_RUN` and asks the user to run the Pascal example. This aggregation applies to the named LLM route and does not redesign the generic or multi-input verifier.

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
