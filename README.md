# TorchVision Builds — CUDA 13.0 Branch (SM110 + SM121)

This branch contains TorchVision build recipes that require **CUDA 13.0+**, needed for SM110 (DRIVE Thor) and SM121 (DGX Spark) GPU architectures.

## Why a Separate Branch?

SM110 (`sm_110`) is not recognized by `nvcc` in CUDA 12.8 or 12.9 — it requires CUDA 13.0+. SM121 (`sm_121`) also benefits from native CUDA 13.0 support, simplifying the build (no special overrides needed).

Since the PyTorch override pattern binds `cudaPackages` from the nixpkgs scope, the only way to use CUDA 13.0 is to pin nixpkgs to a revision that defaults to CUDA 13.0.

## Recipes (12 variants)

### SM110 (NVIDIA DRIVE Thor) — 6 variants

| Package Name | CPU ISA | Platform |
|---|---|---|
| `torchvision-python313-cuda13_0-sm110-avx2` | AVX2 | x86_64-linux |
| `torchvision-python313-cuda13_0-sm110-avx512` | AVX-512 | x86_64-linux |
| `torchvision-python313-cuda13_0-sm110-avx512bf16` | AVX-512 BF16 | x86_64-linux |
| `torchvision-python313-cuda13_0-sm110-avx512vnni` | AVX-512 VNNI | x86_64-linux |
| `torchvision-python313-cuda13_0-sm110-armv8.2` | ARMv8.2-A | aarch64-linux |
| `torchvision-python313-cuda13_0-sm110-armv9` | ARMv9-A | aarch64-linux |

### SM121 (NVIDIA DGX Spark) — 6 variants

| Package Name | CPU ISA | Platform |
|---|---|---|
| `torchvision-python313-cuda13_0-sm121-avx2` | AVX2 | x86_64-linux |
| `torchvision-python313-cuda13_0-sm121-avx512` | AVX-512 | x86_64-linux |
| `torchvision-python313-cuda13_0-sm121-avx512bf16` | AVX-512 BF16 | x86_64-linux |
| `torchvision-python313-cuda13_0-sm121-avx512vnni` | AVX-512 VNNI | x86_64-linux |
| `torchvision-python313-cuda13_0-sm121-armv8.2` | ARMv8.2-A | aarch64-linux |
| `torchvision-python313-cuda13_0-sm121-armv9` | ARMv9-A | aarch64-linux |

## Setup

The nixpkgs pin in each `.nix` file must point to a nixpkgs commit where `cudaPackages` defaults to CUDA 13.0. Update the `url` in `builtins.fetchTarball` accordingly.

## Building

```bash
git checkout cuda-13_0
flox build torchvision-python313-cuda13_0-sm110-avx2
flox build torchvision-python313-cuda13_0-sm121-avx512
```

## Related Branches

- **`main`** — CUDA 12.8 recipes (SM61, SM80–SM100, SM120, CPU)
- **`cuda-12_9`** — CUDA 12.9 recipes (SM103)
