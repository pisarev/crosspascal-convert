# Qwen CUDA first-run result

The Windows x64 converter built the pinned Qwen2.5-0.5B-Instruct profile for the architecture it detected. Two builds of QwenChat used the same resulting bundle.

| Item | Observed value |
| --- | --- |
| Profile | `qwen2.5-0.5b-cuda-f32-64-v1` |
| Model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Tested GPU and target | NVIDIA GeForce RTX 4060 Ti, `sm_89` |
| Pascal compilers | Delphi 13.2 and Lazarus 4.8 with FPC 3.2.2 |
| Model parameters | FP32, all 24 layers, 170 tensors |
| Context, prefill chunk, batch | 64, 64, 1 |
| Prompt | `Say hello in English.` |
| Reply from both builds | `Hello! How can I assist` |
| Generated token IDs | `9707, 0, 2585, 646, 358, 7789` |

The example checked the bundle and runtime identities, ran on CUDA, and stopped at the requested six-token limit. Both processes exited normally.

Manual architecture entry was checked separately in Preview. The GPU shown in the table is the test device. The converter accepts the detected architecture or a manually entered destination architecture for compatible CUDA hardware.

Follow the [Windows installation guide](../INSTALL.md) to build and test your own bundle. The measured result applies to this model, profile, and prompt. It does not measure general language quality or throughput.
