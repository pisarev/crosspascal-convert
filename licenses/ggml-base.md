# ggml-base.dll

Gemma 3n conversion uses the GGML tensor dequantization routines in
`ggml-base.dll`. The Windows x64 binary was taken from the Ollama 0.40.14
archive (`lib/ollama/ggml-base.dll`). Its SHA-256 is
`a63add2beb37c7b55e715028f772e4da828fa7cc6d687f4c3e087c26cdbcce89`.
The [llama.cpp license](LLAMA_CPP_LICENSE.txt) and [vendor notices](LLAMA_CPP_VENDORS_LICENSE.txt)
are the copies shipped with that archive.

The Windows loader also needs the following libraries from the same directory
in the Ollama archive. Keep them next to `ggml-base.dll` in `runtime/converter`.

| Library | SHA-256 |
| --- | --- |
| `libc++.dll` | `8429f6970e0ac54c8566e0a55d64f9fbfeb79173a17dcca32c9a0056f8617684` |
| `libomp.dll` | `69dfa59cf05e8ff7c6b32e2c80e2b86d423f16f21da00203aa1a7da5b6a9fc13` |
| `libunwind.dll` | `814389e7fab11d650c3b4c795a189d8270634647361c31d8582676a13463180f` |
