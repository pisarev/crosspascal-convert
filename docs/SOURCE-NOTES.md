# Converter source notes

This source snapshot shows selected parts of the converter architecture. It is not a complete installable package or a working conversion SDK. Several package and family modules are absent, including dependencies needed by the package entry point. Its documentation and examples describe the included source; they do not establish support for another model, toolchain, or machine.

## Inputs and model contracts

Input names, shapes, and types must come from the selected model. A shape hint uses the last colon as its separator because tensor names may already contain a colon, such as an output suffix from TensorFlow. Unknown input names and unresolved dimensions should be reported explicitly.

File contents matter as well as filenames: the TFLite identifier follows its root-table offset, and a downloaded model may have a generic filename. A directory-based model, an ONNX graph with external tensor files, and a self-contained graph have different dependency requirements. The selected-closure helpers cover only their documented file sets and size limits.

A build recipe records a request. A manifest describes the consumer-facing model contract. Compiler success, valid metadata, and a completed numerical comparison are separate facts. A multi-input model must not receive a single-input manifest, and an adapter's declared output shape must also match its decoding semantics.

## Targets and tuning

The deployment OS, CPU architecture, GPU architecture, and driver requirements are distinct from the build host. Unknown or unsupported combinations must be refused or left for a suitable runner. Compilation on one machine does not prove execution on another.

The selected CPU tuning route uses a local runner, so a compiler target that the local processor cannot execute is not a valid tuning choice. A tuning database must cover the actual model's tasks with measured records; file presence and unmeasured sentinel records do not establish coverage. Historical timings in the source describe specific observations, not a general performance promise.

The cuDNN helper still lacks a complete guard for grouped convolution. Removing explanatory comments does not resolve that limitation or validate CUDA execution.

## File handling and runtime bundles

A checked copy is not an immutable snapshot. Source-freeze and selected-file helpers document the limits of their scans, including changes between observations. Win32 holders keep handles for a defined scope and perform their post-audit before releasing them; their receipts do not establish model accuracy or release-grade immutability.

Private stages and final output names have separate roles. Existing output files must not be overwritten, and a directory name alone does not authorize cleanup of its contents. An atomic namespace commit does not promise durability after power loss.

Runtime inventory ties selected library bytes to the building environment and includes required license files. Untracked libraries in the destination can change what loads. An exact inventory does not prove compatibility with another device, driver, or operating system.

## Comparisons and diagnostics

The included verifier has several comparison profiles with different rules. Output positions, candidate-row associations, and class vectors must retain their documented meaning; a successful comparison under one profile must not be presented as success under another. Synthetic inputs are not a substitute for representative images or task-specific acceptance.

In the generic probe path, matching labels or confidence-threshold decisions are diagnostic observations. They cannot cancel a failed numeric comparison of scores, boxes, or another output. Judging probes also reject invalid shapes, types, empty tensors, and nonfinite values. A deliberately nonjudging noise probe retains its diagnostic role; an output-count mismatch remains a required failure for every probe.

The generic and multi-input profiles still need explicit selection from a complete consumer contract. Some selection and row-matching rules remain heuristic. The outer report also needs consistent separation of an execution error, an unattempted comparison, and a numeric mismatch. These are remaining converter tasks; the separately accepted two-case OmDet decision cannot be applied to arbitrary inputs of this verifier.

A comparison that did not run does not establish acceptance. Native library loading can outlive Python objects, which is why the source isolates certain checks in a child process and checks its actual exit. Do not invoke these paths as an import smoke test.

Thirteen isolated regressions passed for the actual generic comparison helpers using Python 3.12.14 and NumPy 2.3.5 in a separate Linux environment. They cover incorrect boxes, shapes, types, nonfinite values, output counts, failed numeric comparisons with matching labels, and preservation of the noise probe's role. They do not import the package initializer or execute TVM, ONNX Runtime, a model, or the converter CLI.

The sample measurements and support inventory are executable, dated data retained from the source snapshot. Their presence does not update any model's acceptance status.

## Publication cleanup

The initial publication cleanup removed reviewed explanatory line comments while retaining encoding declarations, linter directives, docstrings, diagnostic strings, schemas, and test data. One historical installation path in a docstring was replaced with generic wording. Copyright notices were then added to the 25 component modules; the synthetic test runner was excluded.

The later generic-comparison correction is a behavior change, recorded separately from that text cleanup and checked by the isolated regressions above. This preparation does not run models or certify the package's missing dependency closure. Historical measurements retained in the source describe their original experiments.

