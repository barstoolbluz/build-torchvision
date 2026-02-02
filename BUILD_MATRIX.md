# TorchVision Build Matrix Documentation

## Overview

This document defines the complete build matrix for custom TorchVision builds, explaining all dimensions and their interactions.

**📖 Related Documentation:**
- **[RECIPE_TEMPLATE.md](./RECIPE_TEMPLATE.md)** - Copy-paste templates for generating any variant
- **[README.md](./README.md)** - User guide and quick start
- **[../build-pytorch/BUILD_MATRIX.md](../build-pytorch/BUILD_MATRIX.md)** - PyTorch build matrix (dependency)

**🔗 Dependency:** TorchVision builds require matching PyTorch variants from `../build-pytorch/`

> **Multi-Branch Strategy:** GPU architectures requiring newer CUDA toolkits than the default (12.8) are maintained on dedicated branches. SM103 variants are on the `cuda-12_9` branch (requires CUDA 12.9), and SM110 variants are on the `cuda-13_0` branch (requires CUDA 13.0). The `main` branch carries all architectures compatible with CUDA 12.8.

## Matrix Dimensions

Our build matrix has **4 independent dimensions** (same as PyTorch):

```
Build Variant = f(Python_Version, GPU_Architecture, CPU_ISA, CUDA_Toolkit)
```

### 1. Python Version

**Supported Versions:**
- Python 3.13 (current)
- Python 3.12 (planned)
- Python 3.11 (planned)

**Naming:** `python313`, `python312`, `python311`

**Example:** `torchvision-python313-...`

### 2. GPU Architecture (Compute Capability)

**Supported GPU Architectures:**

| SM Version | Architecture | GPUs | CUDA Requirement | Pattern | Status |
|------------|--------------|------|------------------|---------|--------|
| **SM121** | DGX Spark | DGX Spark (Specialized Datacenter) | CUDA 12.9+ | Type A | Moved to `cuda-13_0` branch (requires CUDA 12.9+) |
| **SM120** | Blackwell | RTX 5090 | CUDA 12.8+ | Type B | ✅ Done (6/6) |
| **SM110** | Automotive | DRIVE Thor, Orin+ | CUDA 13.0+ | Type A | Moved to `cuda-13_0` branch (requires CUDA 13.0) |
| **SM103** | Blackwell DC | B300 (Datacenter) | CUDA 12.9+ | Type A | Moved to `cuda-12_9` branch (requires CUDA 12.9) |
| **SM100** | Blackwell DC | B100/B200 (Datacenter) | CUDA 12.0+ | Type A | ✅ Done (6/6) |
| **SM90** | Hopper | H100, H200, L40S | CUDA 12.0+ | Type A | ✅ Done (6/6) |
| **SM89** | Ada Lovelace | RTX 4090, L4, L40 | CUDA 11.8+ | Type A | ✅ Done (6/6) |
| **SM86** | Ampere | RTX 3090, A5000, A40 | CUDA 11.1+ | Type B | ✅ Done (6/6) |
| **SM80** | Ampere DC | A100, A30 | CUDA 11.0+ | Type A | ✅ Done (6/6) |
| **SM61** | Pascal | GTX 1070, 1080, 1080 Ti | CUDA 11.0+ | Type C | ✅ Done (1 — AVX) |
| **CPU** | None | N/A | N/A | N/A | ✅ Done (6/6) |

**Naming:** `sm121`, `sm120`, `sm110`, `sm103`, `sm100`, `sm90`, `sm89`, `sm86`, `sm80`, `sm61`, `cpu`

**Example:** `torchvision-python313-cuda12_8-sm120-...`

**Important Notes:**
- **Pattern Type A (SM121, SM100, SM90, SM89, SM80; also SM110/SM103 on their respective branches):** Uses `gpuArchSM = "sm_XXX"` and `gpuTargets = [ gpuArchSM ]`
- **Pattern Type B (SM120, SM86):** Uses `gpuArchNum = "X.X"` (decimal) and `gpuTargets = [ gpuArchNum ]` (NO gpuArchSM)
- **Pattern Type C (SM61 and older):** Uses `gpuArchSM = "6.1"` (dot notation, NOT sm_61) and `gpuTargets = [ gpuArchSM ]`
- Always check the corresponding PyTorch pattern before creating TorchVision variants!
- SM120 requires PyTorch 2.7+ with CUDA 12.8+ and driver 570+

### 3. CPU Instruction Set Architecture (ISA)

**x86-64 Optimizations:**

| ISA | Compiler Flags | Hardware | Performance Gain | Compatibility |
|-----|----------------|----------|------------------|---------------|
| **AVX-512 BF16** | `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512bf16 -mfma` | Intel Cooper Lake+ (2020), AMD Zen 4+ (2022) | ~2.5x over baseline (BF16) | Limited |
| **AVX-512 VNNI** | `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512vnni -mfma` | Intel Skylake-SP+ (2017), AMD Zen 4+ (2022) | ~2.2x over baseline (INT8) | Limited |
| **AVX-512** | `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma` | Intel Skylake-X+ (2017), AMD Zen 4+ (2022) | ~2x over baseline | Limited |
| **AVX2** | `-mavx2 -mfma -mf16c` | Intel Haswell+ (2013+), AMD Zen 1+ (2017) | ~1.5x over baseline | Broad |
| **AVX** | `-mavx -mno-fma -mno-bmi -mno-bmi2 -mno-avx2` | Intel Sandy Bridge+ (2011+), AMD Bulldozer+ (2011+) | Baseline | Maximum |

**ARM Optimizations:**

| ISA | Compiler Flags | Hardware | Status |
|-----|----------------|----------|--------|
| **ARMv9** | `-march=armv9-a+sve2` | Neoverse V1/V2, Cortex-X2+, Graviton3+ | ✅ Implemented (SM121, SM120) |
| **ARMv8.2** | `-march=armv8.2-a+fp16+dotprod` | Neoverse N1, Cortex-A75+, Graviton2 | ✅ Implemented (SM121, SM120) |

**Naming:** `avx512bf16`, `avx512vnni`, `avx512`, `avx2`, `avx`, `armv9`, `armv8.2`

**Example:** `torchvision-python313-cuda12_8-sm120-avx512`

### 4. CUDA Toolkit Version

**CUDA Toolkit Compatibility Table:**

| CUDA Version | Min Driver (Linux) | SM121 | SM120 | SM110 | SM90 | SM89 | SM86 | Notes |
|--------------|-------------------|-------|-------|-------|------|------|------|-------|
| **13.0** | 580+ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Latest, removes SM 5.x-7.0 |
| **12.9** | 575+ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Latest 12.x |
| **12.8** | 570+ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | PyTorch 2.7 default |
| **12.6** | 560+ | ✅** | ✅** | ✅ | ✅ | ✅ | ✅ | Stable |

**Current Implementation:** CUDA 12.8 (default)

**Naming:** Package names currently use `cuda12_8` in the middle (e.g., `torchvision-python313-cuda12_8-sm120-avx512`)

## Current Implementation Status

### Total Build Matrix

**Total implemented variants (main branch):** 49
- **GPU builds (SM80–SM121, excluding SM103/SM110):** 42 (7 GPU architectures × 6 CPU ISAs)
  - x86-64: 28 (7 GPU × 4 CPU ISAs)
  - ARM: 14 (7 GPU × 2 CPU ISAs)
- **GPU build (SM61 Pascal):** 1 (AVX only)
- **CPU-only builds:** 6 (6 CPU ISAs)

**Additionally on other branches:**
- **SM103:** 6 variants on `cuda-12_9` branch
- **SM110:** 6 variants on `cuda-13_0` branch

**Currently implemented on main:** 49 ✅

### Implemented Variants

#### SM121 Variants (6/6) ✅ COMPLETE

All SM121 variants use **Pattern Type A** (with gpuArchSM):

| Package Name | CPU ISA | Platform | Status |
|-------------|---------|----------|--------|
| `torchvision-python313-cuda12_8-sm121-avx2` | AVX2 | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm121-avx512` | AVX-512 | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm121-avx512bf16` | AVX-512 BF16 | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm121-avx512vnni` | AVX-512 VNNI | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm121-armv8.2` | ARMv8.2-A | aarch64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm121-armv9` | ARMv9-A | aarch64-linux | ✅ Created |

**GPU Pattern (SM121):**
```nix
gpuArchNum = "121";
gpuArchSM = "sm_121";
gpuTargets = [ gpuArchSM ];
```

#### SM120 Variants (6/6) ✅ COMPLETE

All SM120 variants use **Pattern Type B** (without gpuArchSM):

| Package Name | CPU ISA | Platform | Status |
|-------------|---------|----------|--------|
| `torchvision-python313-cuda12_8-sm120-avx2` | AVX2 | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm120-avx512` | AVX-512 | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm120-avx512bf16` | AVX-512 BF16 | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm120-avx512vnni` | AVX-512 VNNI | x86_64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm120-armv8.2` | ARMv8.2-A | aarch64-linux | ✅ Created |
| `torchvision-python313-cuda12_8-sm120-armv9` | ARMv9-A | aarch64-linux | ✅ Created |

**GPU Pattern (SM120):**
```nix
# PyTorch's CMake accepts numeric format (12.0) not sm_120
gpuArchNum = "12.0";
# NO gpuArchSM variable
gpuTargets = [ gpuArchNum ];
```

### Additional Implemented Variants ✅ COMPLETE

#### SM110 Variants — Moved to `cuda-13_0` branch

**GPU:** NVIDIA DRIVE Thor, Orin+ (Automotive)
**Status:** 6 variants moved to `cuda-13_0` branch (requires CUDA 13.0).

#### SM103 Variants — Moved to `cuda-12_9` branch

**GPU:** NVIDIA Blackwell B300 (Datacenter)
**Status:** 6 variants moved to `cuda-12_9` branch (requires CUDA 12.9).

#### SM100 Variants (6/6) ✅ COMPLETE - Pattern Type A

**GPU:** NVIDIA Blackwell B100/B200 (Datacenter)
**Pattern:** sm_XXX format (`gpuArchSM = "sm_100"`)

- `torchvision-python313-cuda12_8-sm100-avx2` (x86_64)
- `torchvision-python313-cuda12_8-sm100-avx512` (x86_64)
- `torchvision-python313-cuda12_8-sm100-avx512bf16` (x86_64)
- `torchvision-python313-cuda12_8-sm100-avx512vnni` (x86_64)
- `torchvision-python313-cuda12_8-sm100-armv8.2` (aarch64)
- `torchvision-python313-cuda12_8-sm100-armv9` (aarch64)

#### SM90 Variants (6/6) ✅ COMPLETE - Pattern Type A

**GPU:** NVIDIA Hopper (H100, L40S)
**Pattern:** sm_XXX format (`gpuArchSM = "sm_90"`)

- `torchvision-python313-cuda12_8-sm90-avx2` (x86_64)
- `torchvision-python313-cuda12_8-sm90-avx512` (x86_64)
- `torchvision-python313-cuda12_8-sm90-avx512bf16` (x86_64)
- `torchvision-python313-cuda12_8-sm90-avx512vnni` (x86_64)
- `torchvision-python313-cuda12_8-sm90-armv8.2` (aarch64)
- `torchvision-python313-cuda12_8-sm90-armv9` (aarch64)

#### SM89 Variants (6/6) ✅ COMPLETE - Pattern Type A

**GPU:** NVIDIA Ada Lovelace (RTX 4090, L40)
**Pattern:** sm_XXX format (`gpuArchSM = "sm_89"`)

- `torchvision-python313-cuda12_8-sm89-avx2` (x86_64)
- `torchvision-python313-cuda12_8-sm89-avx512` (x86_64)
- `torchvision-python313-cuda12_8-sm89-avx512bf16` (x86_64)
- `torchvision-python313-cuda12_8-sm89-avx512vnni` (x86_64)
- `torchvision-python313-cuda12_8-sm89-armv8.2` (aarch64)
- `torchvision-python313-cuda12_8-sm89-armv9` (aarch64)

#### SM86 Variants (6/6) ✅ COMPLETE - Pattern Type B

**GPU:** NVIDIA Ampere (RTX 3090, A40, A5000)
**Pattern:** Decimal format (`gpuArchNum = "8.6"`)

- `torchvision-python313-cuda12_8-sm86-avx2` (x86_64)
- `torchvision-python313-cuda12_8-sm86-avx512` (x86_64)
- `torchvision-python313-cuda12_8-sm86-avx512bf16` (x86_64)
- `torchvision-python313-cuda12_8-sm86-avx512vnni` (x86_64)
- `torchvision-python313-cuda12_8-sm86-armv8.2` (aarch64)
- `torchvision-python313-cuda12_8-sm86-armv9` (aarch64)

#### SM80 Variants (6/6) ✅ COMPLETE - Pattern Type A

**GPU:** NVIDIA Ampere Datacenter (A100, A30)
**Pattern:** sm_XXX format (`gpuArchSM = "sm_80"`)

- `torchvision-python313-cuda12_8-sm80-avx2` (x86_64)
- `torchvision-python313-cuda12_8-sm80-avx512` (x86_64)
- `torchvision-python313-cuda12_8-sm80-avx512bf16` (x86_64)
- `torchvision-python313-cuda12_8-sm80-avx512vnni` (x86_64)
- `torchvision-python313-cuda12_8-sm80-armv8.2` (aarch64)
- `torchvision-python313-cuda12_8-sm80-armv9` (aarch64)

#### SM61 Variant (1) ✅ - Pattern Type C

**GPU:** NVIDIA Pascal (GTX 1070, 1080, 1080 Ti)
**Pattern:** Dot notation (`gpuArchSM = "6.1"`)
**Note:** cuDNN 9.11+ dropped support for SM < 7.5; NNPACK requires AVX2+FMA3. Both are disabled for SM61-AVX builds.

- `torchvision-python313-cuda12_8-sm61-avx-no-nnpack` (x86_64)

**GPU Pattern (SM61):**
```nix
gpuArchNum = "61";
gpuArchSM = "6.1";          # Dot notation, NOT "sm_61"
gpuTargets = [ gpuArchSM ];  # Passes "6.1"
```

#### CPU-only Variants (6/6) ✅ COMPLETE

- `torchvision-python313-cpu-avx2` (x86_64 or both platforms)
- `torchvision-python313-cpu-avx512` (x86_64 or both platforms)
- `torchvision-python313-cpu-avx512bf16` (x86_64 or both platforms)
- `torchvision-python313-cpu-avx512vnni` (x86_64 or both platforms)
- `torchvision-python313-cpu-armv8.2` (aarch64 or both platforms)
- `torchvision-python313-cpu-armv9` (aarch64 or both platforms)

## GPU Architecture Patterns (CRITICAL!)

TorchVision must match the GPU architecture pattern used by the corresponding PyTorch build. There are **THREE different patterns**:

### Pattern Type A: sm_XXX format (SM121, SM100, SM90, SM89, SM80; SM110/SM103 on other branches)

**Used by:** SM121, SM100, SM90, SM89, SM80 (on main); SM110 (`cuda-13_0` branch), SM103 (`cuda-12_9` branch)

```nix
gpuArchNum = "121";        # For CMAKE_CUDA_ARCHITECTURES
gpuArchSM = "sm_121";      # For TORCH_CUDA_ARCH_LIST
gpuTargets = [ gpuArchSM ]; # Uses sm_121
```

**Verification command:**
```bash
grep -E "gpuArchNum|gpuArchSM|gpuTargets" ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm121-*.nix | head -5
```

### Pattern Type B: Decimal format without gpuArchSM (SM120, SM86)

**Used by:** SM120, SM86 only

```nix
# PyTorch's CMake accepts numeric format for SM120 and SM86
gpuArchNum = "12.0";       # Or "8.6" for SM86
# NO gpuArchSM variable
gpuTargets = [ gpuArchNum ]; # Uses numeric format directly
```

**Verification command:**
```bash
grep -E "gpuArchNum|gpuArchSM|gpuTargets" ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm120-*.nix | head -5
```

### Pattern Type C: Dot notation in gpuArchSM (SM61 and older architectures)

**Used by:** SM61 (and likely other pre-SM80 architectures like SM70, SM75)

```nix
# Older architectures require dot notation in TORCH_CUDA_ARCH_LIST
gpuArchNum = "61";
gpuArchSM = "6.1";         # Dot notation, NOT "sm_61"
gpuTargets = [ gpuArchSM ]; # Uses "6.1"
```

**Why this differs:** PyTorch's CMake/TORCH_CUDA_ARCH_LIST expects dot notation (e.g., `"6.1"`, `"7.0"`, `"7.5"`) for older compute capabilities. The `sm_XXX` format (Type A) is used by newer architectures (SM80+).

**cuDNN/NNPACK caveat:** cuDNN 9.11+ dropped support for SM < 7.5; NNPACK requires AVX2+FMA3. Both are disabled for SM61-AVX builds. Core PyTorch/TorchVision operations work normally.

### Before Creating Variants: ALWAYS CHECK PATTERN!

**CRITICAL:** Before creating TorchVision variants for a new GPU architecture:

1. Check the PyTorch pattern first:
   ```bash
   grep -E "gpuArchNum|gpuArchSM|gpuTargets" ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm{ARCH}-*.nix | head -5
   ```

2. If you see `gpuArchSM` with `sm_` prefix (e.g. `"sm_121"`) → Use Pattern Type A template
3. If you see only `gpuArchNum` (no `gpuArchSM`) → Use Pattern Type B template
4. If you see `gpuArchSM` with dot notation (e.g. `"6.1"`) → Use Pattern Type C template

**Reference:** See `/tmp/gpu-arch-pattern-reference.md` for detailed pattern documentation.

## User Selection Guide

### For Users: How to Choose Your Build

1. **Check your GPU:**
   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   ```

2. **Check your driver:**
   ```bash
   nvidia-smi | grep "Driver Version"
   ```

3. **Match to build variant:**

   | Your GPU | Your Driver | Install This |
   |----------|-------------|--------------|
   | DGX Spark | 570+ | `torchvision-python313-cuda12_8-sm121-avx512` (or other ISA) |
   | RTX 5090 | 570+ | `torchvision-python313-cuda12_8-sm120-avx512` (or other ISA) |
   | RTX 5090 | < 570 | Upgrade driver first! |
   | H100/L40S | 570+ | `torchvision-python313-cuda12_8-sm90-avx512` |
   | RTX 4090 | 570+ | `torchvision-python313-cuda12_8-sm89-avx512` |
   | RTX 3090 | 570+ | `torchvision-python313-cuda12_8-sm86-avx2` |
   | A100 | 570+ | `torchvision-python313-cuda12_8-sm80-avx512` |
   | GTX 1070/1080/1080 Ti | 390+ | `torchvision-python313-cuda12_8-sm61-avx-no-nnpack` (cuDNN+NNPACK disabled) |
   | No GPU | Any | `torchvision-python313-cpu-avx2` |

4. **Check CPU capabilities (for AVX-512 builds):**
   ```bash
   lscpu | grep -E 'avx512|avx2'
   ```

   If no AVX-512: Use AVX2 variant instead.

5. **Ensure matching PyTorch is installed:**
   ```bash
   # TorchVision requires matching PyTorch variant
   flox install yourorg/pytorch-python313-cuda12_8-sm120-avx512
   flox install yourorg/torchvision-python313-cuda12_8-sm120-avx512
   ```

## Build Strategy

### Recommended Approach

Generate variants in this order:

1. **SM120 variants** (RTX 5090) - ✅ DONE
2. **SM121 variants** (DGX Spark) - ✅ DONE
3. **SM90 variants** (H100) - Next priority for datacenter
4. **SM89 variants** (RTX 4090) - Common gaming/workstation
5. **SM86 variants** (RTX 3090) - Still widely used
6. **SM80 variants** (A100) - Datacenter standard
7. **SM110/SM103/SM100** - Specialized/newer datacenter
8. **CPU-only variants** - For development/testing
9. **SM61 variant** (GTX 1070/1080 Ti) - ✅ DONE

### Generation Workflow

For each GPU architecture:

1. **Check PyTorch pattern:**
   ```bash
   grep -E "gpuArchNum|gpuArchSM|gpuTargets" ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm{ARCH}-*.nix | head -5
   ```

2. **Use RECIPE_TEMPLATE.md** to generate 6 variants:
   - 4 x86-64 variants: avx2, avx512, avx512bf16, avx512vnni
   - 2 ARM variants: armv8.2, armv9

3. **Verify pattern matches PyTorch**

4. **Build and test** (optional but recommended)

5. **Commit to git**

## Build Time Considerations

### Build Duration by Variant Type

| Variant Type | Estimated Time | Reason |
|--------------|----------------|--------|
| CPU-only | 30-60 minutes | Fewer compilation targets, depends on PyTorch |
| GPU (single SM) | 1-2 hours | CUDA compilation overhead, builds on top of PyTorch |

**Note:** TorchVision builds are faster than PyTorch builds because they're smaller packages.

### Disk Space Requirements

| Component | Size per Build |
|-----------|----------------|
| Build artifacts | ~5-8 GB |
| Final package | ~500 MB - 1 GB |
| **Total per variant** | ~6-9 GB |

**Recommendation:** Build on CI/CD with at least 100GB free space to accommodate multiple variants.

## PyTorch Dependency Resolution (TODO)

**Current Status:** TorchVision variants use `pytorch.override` to create a custom PyTorch with matching GPU/CPU configuration.

**Future Work:** Set up proper dependency resolution to reference actual PyTorch packages from `../build-pytorch` instead of using `pytorch.override`. This will ensure exact matching between TorchVision and PyTorch builds.

**Implementation Notes:**
```nix
# Current (uses override):
customPytorch = (python3Packages.pytorch.override {
  cudaSupport = true;
  gpuTargets = [ gpuArchNum ];
}).overrideAttrs (oldAttrs: { ... });

# Future (reference actual package):
customPytorch = inputs.build-pytorch.packages.{system}.pytorch-python313-cuda12_8-sm120-avx512;
```

## Summary

**Current Status:**
- ✅ SM121: 6/6 variants created (100%)
- ✅ SM120: 6/6 variants created (100%)
- ➡️ SM110: Moved to `cuda-13_0` branch (6 variants, requires CUDA 13.0)
- ➡️ SM103: Moved to `cuda-12_9` branch (6 variants, requires CUDA 12.9)
- ✅ SM100: 6/6 variants created (100%)
- ✅ SM90: 6/6 variants created (100%)
- ✅ SM89: 6/6 variants created (100%)
- ✅ SM86: 6/6 variants created (100%)
- ✅ SM80: 6/6 variants created (100%)
- ✅ SM61: 1 variant created (AVX)
- ✅ CPU-only: 6/6 variants created (100%)

**Total Progress (main branch): 49 variants ✅** (+ 12 on CUDA-version branches)

**Next Steps:**
1. ✅ 49 variants on main, 12 on CUDA-version branches
2. Test variants on target hardware
3. Set up proper PyTorch dependency resolution
4. Publish to FloxHub for team distribution
5. Document testing procedures and validation results

**Pattern Reference:**
- **Pattern Type A** (gpuArchSM = "sm_XXX"): SM121, SM100, SM90, SM89, SM80 (main); SM110 (`cuda-13_0`), SM103 (`cuda-12_9`)
- **Pattern Type B** (gpuArchNum = "X.X", no gpuArchSM): SM120, SM86
- **Pattern Type C** (gpuArchSM = "X.X" dot notation): SM61 (and older pre-SM80 architectures)
- Always verify PyTorch pattern before creating variants!
