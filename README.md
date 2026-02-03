# TorchVision Builds — CUDA 12.9 Branch (SM103)

This branch contains TorchVision build recipes that require **CUDA 12.9+**, which is needed for SM103 (Blackwell B300) GPU architecture.

## Why a Separate Branch?

SM103 (`sm_103`) is not recognized by `nvcc` in CUDA 12.8 (the default in nixpkgs on `main`). CUDA 12.9 adds SM103 support. Since the PyTorch override pattern binds `cudaPackages` from the nixpkgs scope, the only way to use CUDA 12.9 is to pin nixpkgs to a revision that defaults to CUDA 12.9.

## Version Matrix

| Branch | TorchVision | PyTorch | CUDA | cuDNN | Python | Min Driver | Nixpkgs Pin |
|--------|-------------|---------|------|-------|--------|------------|-------------|
| `main` | 0.23.0 | 2.8.0 | 12.8 | 9.x | 3.13 | 550+ | [`fe5e41d`](https://github.com/NixOS/nixpkgs/tree/fe5e41d7ffc0421f0913e8472ce6238ed0daf8e3) |
| **`cuda-12_9`** ⬅️ | **0.24.0** | **2.9.1** | **12.9.1** | **9.13.0** | **3.13** | **550+** | [**`6a030d5`**](https://github.com/NixOS/nixpkgs/tree/6a030d535719c5190187c4cec156f335e95e3211) |
| `cuda-13_0` | TBD | 2.10 | 13.0 | TBD | 3.13 | 570+ | TBD |

## Recipes (6 variants)

| Package Name | CPU ISA | Platform |
|---|---|---|
| `torchvision-python313-cuda12_9-sm103-avx2` | AVX2 | x86_64-linux |
| `torchvision-python313-cuda12_9-sm103-avx512` | AVX-512 | x86_64-linux |
| `torchvision-python313-cuda12_9-sm103-avx512bf16` | AVX-512 BF16 | x86_64-linux |
| `torchvision-python313-cuda12_9-sm103-avx512vnni` | AVX-512 VNNI | x86_64-linux |
| `torchvision-python313-cuda12_9-sm103-armv8.2` | ARMv8.2-A | aarch64-linux |
| `torchvision-python313-cuda12_9-sm103-armv9` | ARMv9-A | aarch64-linux |

## Setup

The nixpkgs pin in each `.nix` file must point to a nixpkgs commit where `cudaPackages` defaults to CUDA 12.9. Update the `url` in `builtins.fetchTarball` accordingly.

## Building

```bash
git checkout cuda-12_9
flox build torchvision-python313-cuda12_9-sm103-avx2
```

## Related Branches

- **`main`** — CUDA 12.8 recipes (SM61, SM80–SM100, SM120, SM121, CPU)
- **`cuda-13_0`** — CUDA 13.0 recipes (SM110, SM121)
