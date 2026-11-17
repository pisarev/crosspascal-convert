# Huihui Gemma 3n E4B on Windows

The Huihui profiles run the community abliterated Gemma 3n E4B checkpoint
through native TVM with text and image input. CPU and CUDA profiles use the same pinned
GGUF source:

* Checkpoint: `huihui-ai/Huihui-gemma-3n-E4B-it-abliterated`, revision
  `41baa2bf7d97a5a7e18a4943a5a896e9f4cea529`.
* GGUF: `bartowski/huihui-ai_Huihui-gemma-3n-E4B-it-abliterated-GGUF`, revision
  `19d84eb522d97e9e16eecaf49662f9f84404eefa`.
* File: `huihui-ai_Huihui-gemma-3n-E4B-it-abliterated-Q8_0.gguf`.
* SHA-256: `4980dba55c9d9892a99d527a4ebb791f54996abdb93783f8e2d9cbec381ee35d`.
* Image projector: `Anthonyg5005/gemma-3n-e4b-mmproj-gguf`, revision
  `0f9947517b54ab73db110da2664d8f929c256330`, file `gemma-3n-mmproj.gguf`.
* [Pinned projector download](https://huggingface.co/Anthonyg5005/gemma-3n-e4b-mmproj-gguf/resolve/0f9947517b54ab73db110da2664d8f929c256330/gemma-3n-mmproj.gguf).
* Projector SHA-256: `ea48671680ccc6a0e5a2a81815c564cb9594cf1eec0cc3c792498bc476bf036a`
  (1,967,809,440 bytes).

The source contains Q8_0, BF16, and FP32 tensors. The CPU profile decodes
its parameters to FP32. CUDA keeps the original Q8_0 signed blocks and
their block scales, with FP32 activations, host embeddings, and auxiliary
matrices. Model folders include the source attribution, modification
notice, Gemma Terms of Use, and incorporated use policy.

## Prepare and run

1. Download the dedicated Huihui Gemma CPU or GPU Prepare kit from the
   [EA023 release](https://github.com/pisarev/crosspascal/releases/tag/early-access-023).
2. Extract the ZIP to a writable folder and run **Prepare.cmd**.
3. Open the printed `GemmaChat.dproj` in Delphi or `GemmaChat.lpi` in Lazarus.
   Build for Windows 64-bit and run the project. The prepared folder also
   contains a ready Lazarus executable.
4. Click **Load model**, enter a message, optionally attach an image, and
   click **Generate**.
   **Stop** cancels generation; **Unload** releases the model.

The ready context is 4,096 tokens, including the chat template, image
tokens, prompt, and answer. Images use the 768 × 768 profile with 256 image
tokens. The CUDA package contains code for `sm_75`, `sm_86`, `sm_89`,
and `sm_120`. Prepare checks the installed GPU before selecting CUDA.

## Convert the source GGUF

Open **CrossPascalConvert.exe** in the SDK and select:

* **Huihui Gemma 3n E4B abliterated text | CPU**, or
* **Huihui Gemma 3n E4B abliterated text | CUDA**.

Select the pinned GGUF, the matching `gemma3nv` projector GGUF for image
input, and a new output folder. The selected profile uses
its own source recipe and attribution files. The original Gemma profiles
keep their existing recipes. For CUDA, the target starts with the installed
device's compute capability and remains editable for cross-compilation.
Click **Build**, then open the resulting model folder in GemmaChat.

For a custom SDK layout, `HUIHUI_GEMMA3N_PROFILE` in
`runtime/converter/toolchain.ini` points to this recipe directory. Its
default is `profiles/gemma3n-huihui-abliterated` under `runtime/converter`.

The model files use the included Gemma terms. CrossPascal's software
license and the licenses of its dependencies remain separate.
