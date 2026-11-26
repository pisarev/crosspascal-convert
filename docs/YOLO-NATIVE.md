# Native YOLO conversion

`native/ConvertYolo.dpr` builds a model bundle directly from an original `.pt`
checkpoint and a pinned, weight-free ONNX graph recipe. The conversion code is
Object Pascal. It reads tensor storage without executing checkpoint Python code;
Python, PyTorch and an ONNX exporter are not invoked by this command.

The catalog uses `crosspascal.yolo-profiles.v1`. A profile supplies class labels,
input and output contracts, preprocessing, coordinate rules and postprocessing.
The same command handles classification, detection, instance segmentation, pose,
oriented detection, semantic segmentation and depth estimation. Families and
model sizes are selected by catalog entries rather than executable name checks.

The Windows SDK supplies weight-free recipes and manifests for 115 profiles.
Size kits select `n`, `s`, `m`, `b`, `l`, `x`, or `all` in the YoloModels
example. The desktop converter uses the complete catalog independently of
the prepared size. For `ConvertYolo`, set the request file's `catalog` field
to `profiles-all.json` to select from all 115 profiles. The unchanged
`profiles.json` catalog describes the 23 default `n` profiles.

Build the program with `TVM_LAZY_INIT` and the sibling CrossPascal TVM units on
the unit search path. The compiler kit and the consumer runtime are distinct:
the application that opens a finished bundle does not need the compiler library.

Run:

```text
ConvertYolo request.json --validate
ConvertYolo request.json
```

Validation checks the checkpoint and recipe SHA-256, binds the original weights,
compares every graph input/output name, shape and dtype with the catalog, and
checks the pinned consumer runtime files. It creates no output directory.
Conversion performs those checks, compiles the graph, links its library and
installs a new bundle directory. An existing destination is never replaced.

Example Linux request:

```json
{
  "schema": "crosspascal.yolo-convert.v1",
  "catalog": "profiles-all.json",
  "profile": "yolo26n-seg",
  "checkpoint": "weights/yolo26n-seg.pt",
  "recipe": "recipes/yolo26n-seg.recipe.onnx",
  "destination": "bundles/yolo26n-seg-cpu",
  "device": "cpu",
  "compiler": {
    "directory": "toolchain/compiler-runtime",
    "linker": "/usr/lib/llvm-17/bin/clang++",
    "ffi_library": "toolchain/compiler-runtime/libtvm_ffi.so"
  },
  "runtime": {
    "directory": "runtime",
    "manifest_sha256": "<SHA-256 of runtime/runtime.json>",
    "manifest_bytes": 1046
  }
}
```

Paths are absolute or relative to the request file, including paths inside nested
objects. The parent of `destination` must exist. Supply the actual runtime manifest
digest and byte count; the example byte count is not a fixed format requirement.

The optional `target_os` field selects `linux` or `windows`; its default is the
compiler host's operating system. A Linux host can build either target. A Windows
host builds Windows libraries.

For a Windows target, `compiler` also contains the `msvc_libraries`,
`ucrt_libraries` and `windows_libraries` directories. Its `linker` points to
`link.exe` on Windows or `lld-link` on Linux, and `ffi_library` to the Windows
`tvm_ffi.lib` import library. The compiler directory still contains libraries for
the build host. The consumer runtime contains the Windows DLLs described by its
`runtime.json`; file names, hashes, PE architecture and ABI metadata are checked
for the selected target. Cross-compilation does not load Windows DLLs on Linux.

For CUDA, set `device` to `cuda` and add:

```json
{
  "cuda": {
    "nvrtc": "toolchain/cuda/lib/libnvrtc.so.12",
    "architecture": "sm_89"
  }
}
```

Choose `sm_75`, `sm_86`, `sm_89` or `sm_120` for the intended GPU. This is an
explicit compilation target. It is not inferred from the build host's GPU name.
The consumer checks the actual GPU's compute capability before loading a model.
CPU and CUDA host code use AVX2/FMA with enabled operating-system AVX state.

A finished `crosspascal.yolo-bundle.v1` directory contains `manifest.json` and
`model.so` (Linux) or `model.dll` (Windows). The manifest binds the normalized
profile, checkpoint and recipe digests, model file size/digest, runtime manifest
pin, platform, ABI and device target. The command emits a JSON receipt with the
manifest digest and byte count. Conversion errors return exit code 1; invalid
command usage returns 2.

The consumer calls `LoadYoloBundle` from `CrossVision.ML.YoloBundle` with the
expected profile and manifest pin from its trusted application catalog, then
opens the matching typed interface. These external pins establish which model
the application intended to load; an embedded manifest alone is insufficient.

For adjustable box thresholds and cooperative cancellation, use
`TYoloRunOptions` from `TVM.Yolo.RunOptions` with `LoadYoloBundleForRun`:

```pascal
Options := TYoloRunOptions.FromProfile(ExpectedProfile);
Options.Confidence := 0.35;
Options.NmsIoU := 0.55;
Options.ShouldStop := Worker.Stopping;
Bundle := LoadYoloBundleForRun(BundleDirectory, ManifestSHA256, ManifestBytes,
  RuntimeDirectory, ExpectedProfile, Options);
Detector := Bundle.OpenDetector;
Detections := Detector.Detect(Image);
```

Both thresholds accept finite values from zero to one. Confidence controls box
filtering; NMS IoU applies to heads that use NMS. Classification, semantic maps
and depth retain their task-specific output contracts. The trusted catalog
profile establishes artifact identity, while run options configure inference.

The stop callback reads cancellation state on the inference thread. Keep its
owner alive through the bundle and opened models, make the callback thread safe,
and return promptly without re-entering TVM or the model. Cancellation raises
`ECpuGraphStopped`; the caller can handle it and reuse the model for another run.

The existing `ConvertVisionOnnx` entry point and its older CPU vision catalog
remain separate. This command does not change their catalog schema or other
model adapters.
