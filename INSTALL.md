# Install and use CrossPascal Converter

Download and extract the [Windows x64 SDK](https://github.com/pisarev/crosspascal/releases/tag/early-access-021).
The SDK contains the desktop converter, the native TVM compiler and the runtime libraries.
The converter needs the x64 MSVC linker and the Windows SDK libraries. For CUDA
conversion use NVIDIA CUDA Toolkit 12.8 with NVRTC. It can compile both Pascal
GPUs such as the GTX 1060 and Blackwell GPUs such as the RTX 5090. Keep the SDK
folders together. Prepared CUDA inference requires only the compatible driver
and the supplied TVM runtime.

Open `CrossPascalConvert.exe`, select a profile and its original source, and enter a
new output directory. Choose **Preview** to check the toolchain, then **Build**.
Select the generated bundle in GemmaChat, QwenChat, VisionModels, PaddleOCR, or BiRefNet.

| Profile | Input |
| --- | --- |
| Gemma 3n E4B CPU or CUDA | GGUF. For image input, also select the matching projector GGUF. |
| Gemma 2 CPU or CUDA | Safetensors checkpoint folder with configuration and tokenizer. |
| Dense Qwen2.5 CUDA | Safetensors checkpoint folder with configuration and tokenizer. |
| Qwen2.5-VL 3B CPU or CUDA | Supported original Qwen2.5-VL-3B-Instruct Safetensors snapshot. |
| YOLO11n detection and classification on CPU or CUDA | Pinned Ultralytics `.pt` or the matching ONNX. |
| RT-DETR R18vd CPU or CUDA | Pinned PekingU Safetensors or the matching ONNX. |
| SqueezeNet 1.1 and ResNet50 v1 on CPU or CUDA | Original ONNX. |
| PP-OCRv5 Server detection or recognition on CPU or CUDA | Pinned ONNX from the EA020 source model downloads. Keep `ppocrv5_dict.txt` beside the recognizer ONNX. |
| BiRefNet HR800 background removal on CPU or CUDA | Pinned full SwinL ONNX from the EA021 BiRefNet source pack. |

For CUDA, enter the target compute capability as `sm_61`, `sm_75`, `sm_86`,
`sm_89` or `sm_120`. A conversion builds one architecture. The ready EA019
language downloads carry all five variants and share the weight files.

The ready BiRefNet GPU pack targets `sm_89`, checked on an RTX 4060 Ti. Its CPU pack requires AVX2 and FMA. The [BiRefNet example](https://github.com/pisarev/crosspascal/tree/main/examples/birefnet) saves an original-size mask and transparent cutout. The fixed 800-pixel input differs from the upstream 2048-pixel workflow.

Use ASCII paths for the conversion source, output and compiler tools. The ready BiRefNet GPU pack requires at least 4 GiB of free GPU memory.

For a custom toolchain, set the appropriate paths in `runtime/converter/toolchain.ini`
under `[toolchain]`: `COMPILER`, `LINKER`, `FFI_LIB`, `MSVC_LIB`, `UCRT_LIB`,
`WINDOWS_LIB` or `NVRTC`. The converter accepts absolute paths and paths relative
to `runtime/converter`.

## Browser converter in earlier releases

This guide takes you from the Windows x64 SDK to a short Qwen answer in Delphi or Lazarus. It uses the supplied converter and Python environment.

Use the coordinated Windows x64 SDK and the companion assets listed in the release notes. Before starting the converter, run the TVM and Python setup command in the [Windows setup guide](https://github.com/pisarev/crosspascal/blob/main/docs/WINDOWS-EARLY-ACCESS.md) (`crosspascal/docs/WINDOWS-EARLY-ACCESS.md` in the SDK). It downloads and checks the pinned packages in the SDK.

## 1. Check the PC and decide where the files will live

The first-run profile is deliberately specific:

| Item | First-run configuration |
| --- | --- |
| Operating system | 64-bit Windows supported by CUDA Toolkit 13.2 |
| GPU | CUDA-capable NVIDIA device with a supported driver and enough free memory for the selected model |
| NVIDIA driver | A Windows driver that supports CUDA 13.x |
| CUDA Toolkit | 13.2.0; the tested CUDA compiler reports 13.2.51 |
| LLVM tools | 21.1.1, Windows x64, including `clang.exe` and `lld-link.exe` |
| Windows build libraries | Visual Studio 2022 C++ tools, MSVC v143 and Windows SDK |
| Pascal compiler | Delphi 13.2 or Lazarus 4.8 with FPC 3.2.2, Windows x64 target |
| Python | 3.12.10, included in the SDK. Install the pinned packages with the setup command above. |

The converter detects the local CUDA architecture. Keep it for a local build or enter the destination architecture to prepare a bundle for another computer. QwenChat checks the bundle target against the actual device.

For a *compatible but unpinned local Qwen2 dense-text Safetensors folder*, paste the absolute folder path into Model and select **Compatible local Qwen2 (CUDA FP32, 2,048 tokens)**. This route is separate from the two official 0.5B/1.5B downloads; it does not download weights or claim the local model belongs to either official checkpoint. Preview checks the actual configuration, complete source identity, tokenizer and tensor layout and shows conservative disk, RAM and GPU VRAM requirements. Use the supplied Qwen CUDA consumer kit for its native tokenizer and runtime files only. Native QwenChat performs its own bundle/VRAM checks before loading; it is the operator's responsibility to complete the CUDA/Delphi acceptance, which is not supplied by a converter unit test. Unsupported architectures, unrecognized tokenizers, quantization, MoE, vision/audio and model-provided custom code are refused.

The examples in this guide use `C:\CrossPascal`. You may use another writable local folder; substitute that folder in all paths below. Keep the five repository folders together. Use a local NTFS disk for the conversion source and output.

Leave room for the archive, its extracted contents, the source model download (about 1 GB for 0.5B or 3.1 GB for 1.5B), the generated parameters (1.84 GiB or 5.75 GiB respectively), and temporary build files. The model also needs GPU working memory beyond its parameters. Close your own unneeded GPU applications before the first run; the example checks free GPU memory before loading.

## 2. Download and extract the complete CrossPascal bundle

1. Open [CrossPascal Releases](https://github.com/pisarev/crosspascal/releases) in your browser. The source repositories and releases are public; no GitHub sign-in is required for downloads.
2. Open the Windows x64 early access release. Expand **Assets** and download the Windows x64 SDK asset listed in the release notes. The automatically generated **Source code** archives contain repository source only. **Code > Download ZIP** also downloads only one repository.
3. In File Explorer, locate the downloaded archive. Right-click it and choose **Extract All**. Extract the complete contents to your chosen working location.
4. Open the extracted folder. Find the directory containing `CrossPascalConvert.exe` and all five repository folders. If extraction created an extra outer folder, use the inner directory containing those items as your working root, or move its complete contents together to `C:\CrossPascal`.
5. Compare the layout with this table before continuing:

| Path under `C:\CrossPascal` | Purpose |
| --- | --- |
| `CrossPascalConvert.exe` | Opens the converter in your browser |
| `runtime\python\python.exe` | Portable Python for the converter and TVM setup |
| `crosspascal` | Main Pascal components and runnable Delphi and Lazarus examples |
| `crosspascal-convert` | Converter sources and this guide |
| `crosspascal-tvm` | TVM Pascal bindings and selected runtime files |
| `crosspascal-opencv` | OpenCV Pascal sources |
| `crosspascal-leptonica` | Leptonica Pascal sources |
| `crosspascal-tvm\runtime\windows-x64-cuda` | The matching CUDA consumer runtime and notices |

These are separate projects delivered as one coordinated bundle. Their sibling layout gives the Delphi and Lazarus projects usable relative search paths. The Qwen example uses the TVM units and some OpenCV Pascal interface types. Leptonica belongs to the image examples and is not needed to load this language model.

If the launcher, portable Python or runtime directory is absent, obtain the complete Windows asset named in the release notes before proceeding. Collecting isolated files from different releases does not produce a matching installation.

## 3. Install the Windows C++ tools and runtime

The converter generates a native Windows model library using a separate C++ toolchain.

1. Download Microsoft's [Build Tools for Visual Studio 2022 installer](https://aka.ms/vs/17/release/vs_BuildTools.exe) in your browser, then open the downloaded installer.
2. In the installer, select **Desktop development with C++**. In the installation details, keep **MSVC v143 - VS 2022 C++ x64/x86 build tools** and a **Windows SDK** selected. The [official component list](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=vs-2022#desktop-development-with-c) describes this workload.
3. Choose **Install** and wait for completion. If Visual Studio 2022 already has this workload, use **Modify** to check the components instead of creating another installation.
4. Open the [Microsoft Visual C++ Redistributable download page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/). Under the supported v14 package, download **X64**, open `vc_redist.x64.exe` and complete **Install** or **Repair**. The runtime is needed when native libraries are loaded; the runtime installer alone does not install the development tools.
5. Complete any restart requested by the installer before opening the converter.

The converter normally uses the selected LLVM `clang` and its `lld-link`. It can also locate an installed MSVC linker. Keep the Windows SDK and C++ libraries available for native linking; this guide does not assume that LLVM's installer supplies all Microsoft libraries.

## 4. Install LLVM 21.1.1

1. Open the [official LLVM 21.1.1 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-21.1.1).
2. In **Assets**, select **LLVM-21.1.1-win64.exe**. The [direct official download](https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.1/LLVM-21.1.1-win64.exe) is the same installer. Its published size is 374,399,611 bytes.
3. Open the installer. When it offers PATH integration, choose **Add LLVM to the system PATH for the current user** or the equivalent PATH option for your installation.
4. Keep the default destination, `C:\Program Files\LLVM`, unless you need another location. Complete the installation.
5. In File Explorer, open `C:\Program Files\LLVM\bin`. Check that both `clang.exe` and `lld-link.exe` are present. An unrelated `clang.exe` supplied by another application is not a substitute for this matched toolset.

Use 21.1.1 for this first-run profile. The presence of a newer compiler does not establish that the supplied model route was tested with it.

## 5. Install CUDA Toolkit 13.2.0

1. Open [NVIDIA's CUDA Toolkit 13.2.0 archive](https://developer.nvidia.com/cuda-13-2-0-download-archive).
2. Choose **Windows**, **x86_64**, your supported Windows version, then the **exe (local)** installer. Download it in your browser and open it.
3. Follow the installer. Keep the CUDA toolkit, development headers and NVRTC components selected. The complete toolkit supplies the compiler and headers needed during conversion. If you select a custom installation, retain those components.
4. Keep the standard toolkit directory: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`.
5. Keep a compatible NVIDIA driver installed. If the installer offers a driver older than your working compatible driver, keep your existing driver. Use NVIDIA's [driver download page](https://www.nvidia.com/en-us/drivers/) when a driver installation is needed.
6. Complete installation and any requested restart.

NVIDIA's [Windows installation guide for CUDA 13.2](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-microsoft-windows/index.html) lists supported Windows releases and installer components. CUDA 13.2.0 is the toolkit release name; individual tools can report a component version such as 13.2.51.

### If the converter cannot find CUDA or LLVM

The installers normally configure their paths. If detection later reports a missing tool, check them in the Windows interface:

1. Open **Start**, type **Edit system environment variables**, and open that result.
2. On the **Advanced** tab, choose **Environment Variables**.
3. Find `CUDA_PATH`. For the standard installation its value should be `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`. Edit it if it points to a different CUDA installation. If it is missing, create it with **New**.
4. Select the applicable `Path` variable and choose **Edit**. Add `C:\Program Files\LLVM\bin` and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin` as separate entries if missing. Keep the other entries. If another LLVM installation appears earlier, place the selected LLVM 21.1.1 entry ahead of it.
5. Confirm the dialogs with **OK**. If the converter was already open, use **Quit converter**, then open `CrossPascalConvert.exe` again so its new process receives the updated environment.

If you chose different installation directories, enter those actual directories. The Windows user-variable area is sufficient for tools used by that user; use system variables when your installation is managed for all users.

## 6. Open the converter and obtain Qwen

1. In File Explorer, open `C:\CrossPascal\CrossPascalConvert.exe`.
2. The converter opens a local page in your normal browser. Keep that page open during the conversion.
3. On **Step 1: Select a model**, choose **Qwen2.5-0.5B-Instruct** (the tested first-run profile) or **Qwen2.5-1.5B-Instruct**, then choose **Download selected Qwen model**. The official sources are `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775` (about 1 GB), and `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (about 3.1 GB), under Apache 2.0. The converter checks the pinned sizes and SHA-256 hashes, including the license and tokenizer files, rather than trusting folder names.
4. When the download finishes, the model-path field is filled automatically. Check that Recognise reports the **same model size** you selected. For a pre-existing local HF directory, use Recognise and Preview; the named route will reject a different source-tree SHA-256 even if its directory is similarly named.
5. If you already downloaded this exact model through the converter, paste that existing model-folder path into the field and choose **Recognise**. You can reuse it for another conversion; a new download is unnecessary.
6. Choose **Next** to open **Step 2: Detect this computer**. Review the detected GPU, compiler and CUDA tools. For this route the language-model compiler and CUDA must be available. The optional sample-build section is a separate diagnostic and is not part of this first run.
7. If an environment check failed, correct the reported prerequisite and choose **Refresh**. Continue once the required tools are detected.

The source folder contains the original Hugging Face files. The output folder will contain the converted TVM model. They serve different purposes and must remain separate.

## 7. Select the target and output folder

1. In File Explorer, create the parent directory `C:\CrossPascal\Models` if it does not already exist.
2. On **Step 3: Choose the target computer**, select the named Qwen profile. Check these values:

| Control or setting | Value |
| --- | --- |
| Language model profile | `Qwen 0.5B | CUDA | FP32 | 64 tokens` or `Qwen 1.5B | CUDA | FP32 | 64 tokens`, matching the recognized source |
| Device button | **GPU** |
| CUDA architecture | Detected architecture for this computer, or the manually entered destination architecture |
| Target OS | **Windows** |
| CPU architecture | `x86_64` |
| Weight precision | **float32** |
| New output directory | `C:\CrossPascal\Models\qwen-cuda` |

3. Choose a new folder outside the downloaded model directory. Leave the suggested sibling `qwen-cuda` absent until the build. The converter creates it during the build and refuses to overwrite an existing output. A failed or interrupted build may leave a partial child directory; preserve it for diagnosis and choose a different new child name for a later build, such as `qwen-cuda-2`.
4. Expand **Runtime and compiler paths** only to inspect the defaults. **Tokenizer and runtime folder** should point to `C:\CrossPascal\crosspascal-tvm\runtime\windows-x64-cuda`. **LLM toolchain Python** should point to `C:\CrossPascal\runtime\python\python.exe`. Keep **Include tokenizer and runtime** selected. For 1.5B, Preview prepares the model's source record and license from the pinned catalog and installed files. The same TVM setup serves both profiles.
5. Choose **Next**. **Step 4: Model settings** shows the complete model name, named profile, target, precision, all layers (24 for 0.5B or 28 for 1.5B), context and prefill chunk of 64 tokens, and batch size 1. Review them, then choose **Next**.

The 64-token capacity includes the chat template, prompt and generated reply. It is intended for a small first run. Choosing another dtype or context changes the model contract and requires a separately supported profile.

## 8. Review Preview, then build

1. On **Step 5: Review the expected output**, check the full model name, official source revision, CUDA target and final output directory. Check that the toolkit is compatible and that the Qwen profile shows context and prefill chunk `64 / 64` with batch `1`.
2. The expandable **Source identities, planned files and technical details** section retains hashes, planned files and detailed checks. Preview checks whether this build can start; it has not yet generated a reply.
3. Choose **Next** to open **Step 6: Build and export**, then choose **Build**.
4. Wait for compilation, parameter serialization and final file checks. Follow progress on the page. Keep the model and runtime files in place while the operation runs.
5. When the page reports a built language-model bundle, note its output directory and open it in QwenChat after building the example in your IDE. Use **Download ZIP** to copy the completed bundle to another computer.

The per-build **Application verification** result can still contain `NOT_RUN`: the converter has built the files, while the next section runs them in the real consumer. Do not change or remove that status to make a report look complete. A separate tensor comparison, where present for another route, has its own recorded role and result.

If Build fails, read its specific error before trying again. The converter does not turn an incomplete directory into a completed bundle. Correct the cause, review Preview again, and use a new output directory when necessary.

## 9. Open and build QwenChat

**Delphi 13.2**

1. Start Delphi.
2. Choose **File > Open Project**. Open `C:\CrossPascal\crosspascal\examples\qwen-chat\QwenChat.dproj`.
3. In **Project Manager**, activate the **Release** build configuration and **Windows 64-bit** target. If Windows 64-bit is absent, add that platform to your Delphi installation before building.
4. Choose **Project > Build QwenChat** and wait until the Messages window reports a successful build.
5. The executable is `C:\CrossPascal\crosspascal\examples\qwen-chat\bin\Win64\Release\QwenChat.exe`. Run it through Delphi's **Run > Run Without Debugging**, or open that executable in File Explorer.

The supplied `.dproj` already includes the relative TVM and OpenCV Pascal search paths and the required compiler settings. No design-time component installation is needed. The form is created by the Pascal source, so there is no missing `.dfm` file to obtain.

If you inspect **Project > Options > Building > Delphi Compiler**, the search paths include `..\..\..\crosspascal-tvm\pascal` and `..\..\..\crosspascal-opencv\pascal`. Keep `TVM_LAZY_INIT` in the conditional defines and runtime packages disabled. Use the supplied project settings for this first run.

**Lazarus 4.8**

1. Open `QwenChat.lpi` from the same example directory.
2. Build with the Windows x64 FPC 3.2.2 compiler.
3. Run the project or open `bin/lazarus-win64/QwenChat.exe`.

## 10. Generate the first answer on the GPU

1. In QwenChat, set **Converted model bundle** to `C:\CrossPascal\Models\qwen-cuda`. You can paste the path directly into the field or use **Browse**. Select the directory containing `model.json`.
2. **Pinned CUDA runtime directory** should already point to `C:\CrossPascal\crosspascal-tvm\runtime\windows-x64-cuda`. Correct the path if you extracted the bundle elsewhere.
3. Enter `Say hello in English.` in the prompt field.
4. Leave **Max new tokens** at `0` for Auto.
5. Choose **Run on GPU**. The status line reports file validation, parameter loading and generation. Wait for that operation to finish.
6. Read the reported device and response. Confirm that the reported architecture matches the target shown in Preview.
7. To keep your own result, choose **Save result** and select a new JSON filename in the ordinary Save dialog. Existing reports are not overwritten.
8. Close QwenChat normally after its operation finishes. When finished with conversion, choose **Quit converter** in the browser page. Closing its browser tab alone leaves the local converter service running.

A six-token reply may stop in the middle of a sentence. In the recorded run the stop reason was the requested token limit. The complete first-run application check includes correct file identities, the actual CUDA device, finite outputs of the expected shape, an intelligible short reply and normal shutdown. It does not require every internal number to match a different inference engine.

The [first-run validation record](docs/FIRST-RUN-VALIDATION.md) describes the measured result. The example's successful run is specific to its model, profile, prompt and device; it does not certify arbitrary prompts, other models or other GPU architectures. Your new conversion retains its own file hashes and run report.

## 11. Convert a supplied CPU vision model

Use the matching Windows SDK and CPU model source archive from the same release. Build OpenCV and install the TVM runtime using the [Windows setup guide](https://github.com/pisarev/crosspascal/blob/main/docs/WINDOWS-EARLY-ACCESS.md) (`crosspascal/docs/WINDOWS-EARLY-ACCESS.md` in the SDK) before running VisionModels. Follow the table to select the exact input for each profile.

1. In your browser, open [CrossPascal Releases](https://github.com/pisarev/crosspascal/releases) and download the listed assets. No GitHub account or token is required.
2. Download the coordinated CPU model source archive and Windows SDK listed in the release notes. Extract them into separate directories and keep the source-pack folders intact.
3. In Explorer, create `C:\CrossPascal\Sources`. Right-click the source ZIP, choose **Extract All**, and extract its complete contents there. The archive contains a `crosspascal-cpu-model-sources` folder with these model subfolders:

| Model profile in VisionModels | Folder | ONNX file to select |
| --- | --- | --- |
| SqueezeNet 1.1 | `squeezenet` | `squeezenet1.1-7.onnx` |
| ResNet50 | `resnet50` | `resnet50-v1-12.onnx` |
| YOLO11n classification | `yolo11n-cls` | `yolo11n-cls.onnx` |
| YOLO11n detection | `yolo11n` | `yolo11n.onnx` |
| RT-DETR R18vd | `rtdetr-r18vd` | `rtdetr-r18vd.onnx` |

4. For a first example, open `C:\CrossPascal\Sources\crosspascal-cpu-model-sources\resnet50`. Keep its ONNX, `SOURCE.json`, licence, labels and accompanying records together. Open its `README.md` when you want the exact source and preprocessing details.
5. Right-click `resnet50-v1-12.onnx` and choose **Copy as path**. On older Windows, hold **Shift** while right-clicking to reveal that command.
6. Open `C:\CrossPascal\CrossPascalConvert.exe`. On **Step 1: Select a model**, paste the copied path into the local model-path field and click **Recognise**. Explorer's enclosing double quotes are accepted. For this route, use the path field: dropping one file into the browser does not carry its neighbouring source metadata.
7. Wait for recognition. It must show **Source pack**, the intended model/profile, **Declared model licence**, and the exact **Public source** URL. The converter checks the ONNX and all declared accompanying files. If the page instead says **No adjacent SOURCE.json**, return to the extracted model folder and paste its actual local path.
8. Continue through **Step 2: Detect this computer**. The native compiler and Windows linking prerequisites from sections 3 and 4 must be available. Correct a reported missing prerequisite before proceeding.
9. On **Step 3: Choose the target computer**, confirm **CPU**, **Windows**, **x86_64**, and tuning **Off**. The source pack fills the exact input shape. Create the parent `C:\CrossPascal\Models` in Explorer if needed, then enter a new child such as `C:\CrossPascal\Models\resnet50-cpu` as the output directory. Leave that child nonexistent; the converter creates it.
10. On **Step 4: Model settings**, review the filled values. The supplied classifiers use adapter **yolo11** and NCHW. ResNet50 and SqueezeNet use the recorded RGB mean and standard deviation; YOLO11 classification deliberately has both fields empty. RT-DETR uses adapter **rtdetr** and has no YOLO NMS threshold. Leave the exact pack values in place. A changed contract requires recognition/review of an appropriate source profile.
11. On **Step 5: Review the expected output**, confirm the source profile, declared licence, CPU target and new output directory. The technical details include the bound `SOURCE.json` SHA-256 and public source. Preview binds all selected model and ancillary bytes; changing them requires another recognition and Preview.
12. Continue to **Step 6: Build and export** and choose **Build**. Wait for completion. Read any specific failure before retrying; a successful conversion is separate from the image application run below.
13. The output contains the model DLL, its `.inputs` sidecar, runtime files, recipe, report, and the model manifest when its existing required checks pass. It also contains `model-source\SOURCE.json` and every declared licence, label and export record. The manifest and recipe bind that source metadata. The original ONNX stays in its source folder and is not duplicated into `model-source`.
14. **Download ZIP** is optional for running on this PC. When used, it contains the same source metadata and accompanying files; the converter refuses a download if the approved metadata or manifest binding has changed.

The retained licence is the source declaration. For the applicable terms, read the licence and source notices preserved with that model. Original source identity, local replay path, conversion checks and application results remain separate records.

### Open the vision application

1. Open `VisionModels.dproj` in Delphi 13.2 or `VisionModels.lpi` in Lazarus 4.8. Use the Windows x64 target.
2. In Delphi, select **Release** and **Windows 64-bit**, then choose **Project > Build VisionModels**. In Lazarus, build the supplied project with the Windows x64 FPC compiler. Keep the sibling project folders for the relative TVM, OpenCV, and RTL search paths.
3. Run the project from your IDE. Delphi writes the executable to `bin/Win64/Release` under the example. Lazarus writes it to `bin/lazarus-win64`.
4. Select the same model profile you converted, for example **ResNet50**. Set the converted-bundle field to `C:\CrossPascal\Models\resnet50-cpu`, using **Browse** or by pasting the path.
5. Keep the matching TVM runtime directory from this SDK. Its default is `C:\CrossPascal\crosspascal-tvm\runtime\windows-x64-cuda`. This CPU example selects the pinned TVM CPU runtime libraries from that supplied folder; the folder name does not select GPU inference.
6. Choose **Open image** and select a photograph stored on your PC. Choose **Run on CPU** and wait for completion.
7. A classifier displays class IDs and its original scores. A detector displays class IDs, scores and boxes. The supplied label file records the index order; the application does not guess names or relabel all scores as probabilities.
8. Choose **Save result** to write a new JSON report, then close the application normally. Close the converter with **Quit converter** when finished.

The consumer checks the original model identity, profile and actual generated DLL against its manifest. The generated DLL is allowed to have a different hash after a fresh build, provided its own manifest records that exact DLL and the supported source/contract agree. Output-folder names and ordinary PE build timestamps do not require copying an old developer build. Successful execution on one photo does not establish an upstream benchmark accuracy result.

| Source-pack problem | Action |
| --- | --- |
| Only a generic ONNX is recognized | Paste the original extracted ONNX path. Keep its adjacent `SOURCE.json`; single-file browser upload cannot preserve the pack. |
| SOURCE.json, a licence or another file fails its size/hash check | Extract the complete source archive again into a new folder. Keep files from one candidate together. |
| Preview says a setting differs from the source profile | Press **Recognise** to refill that pack's exact CPU settings, then review Preview again. Empty YOLO classification mean/std and RT-DETR's empty NMS are intentional. |
| Source files changed after Preview | Finish arranging the extracted files, recognize the model again and request a fresh Preview. |
| The consumer rejects the source/profile or runtime | Use the matching Windows x64 SDK, source pack, selected profile and generated bundle. Read the exact rejection; an unrelated model with the same architecture name is not interchangeable. |


## Troubleshooting

| What you see | What to check or do |
| --- | --- |
| No `CrossPascalConvert.exe`, portable Python or CUDA runtime folder | Obtain the complete Windows x64 SDK from the release assets and preserve its folder layout. |
| Converter cannot find CUDA or LLVM | Check the installed versions and Windows environment variables in sections 4 and 5. Quit and reopen the converter after changing them. |
| Windows linker, `kernel32.lib`, C runtime or SDK library is missing | Open Visual Studio Installer, modify Build Tools 2022, and install the C++ workload with MSVC v143 and a Windows SDK. Then reopen the converter. |
| A TVM runtime DLL or Python package is missing | Run the TVM setup command linked at the start of this guide from the extracted SDK root. |
| A `VCRUNTIME` or `MSVCP` DLL is missing | Install or repair Microsoft's Visual C++ v14 **X64** Redistributable. Keep the supplied model/runtime DLL set together. |
| Source files or hashes do not match | Use the converter's exact recorded Qwen revision and the matching runtime from the same release. Reacquire an incomplete or damaged source; do not mix individual files from other builds. |
| The output directory already exists | Choose a new child directory under an existing writable parent. Keep the old converted bundle intact. |
| Preview is no longer valid after editing a setting | Return to Preview and review the current source, target and output again before Build. |
| Delphi cannot find a TVM or OpenCV unit | Open the supplied `QwenChat.dproj` and restore the sibling `crosspascal`, `crosspascal-tvm` and `crosspascal-opencv` folders. Check the relative search paths in section 9. |
| Delphi is building Win32 | Activate **Windows 64-bit** and **Release** in Project Manager, then build again. |
| The example rejects the GPU or model profile | Compare the architecture in the bundle with the device shown by the application. Return to the converter and build for that device. |
| The prompt exceeds the context | Shorten the prompt or reduce the requested new tokens. The chat template shares the context with the prompt and response. |
| GPU memory is insufficient | Close your own unneeded GPU applications and retry after they release memory. The model needs working space in addition to its 1.84 GiB (0.5B) or 5.75 GiB (1.5B) of parameters. |
| Build succeeds but application verification says `NOT_RUN` | Complete sections 9 and 10. Compilation and file checks do not themselves execute the Delphi consumer. |
| A runtime-version or integrity check fails | Keep the converter, model bundle, Pascal sources and runtime from the same matched release. The example deliberately refuses incompatible combinations. Report the exact error and versions if the supplied combination fails. |

For image and document examples, continue with the matching instructions in the main CrossPascal, OpenCV and Leptonica projects. Their availability and validation scope are listed in the same release notes; their source presence alone does not extend this Qwen profile.
