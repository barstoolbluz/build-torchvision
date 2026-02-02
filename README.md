# TorchVision Architecture-Specific Builds

Build custom TorchVision variants optimized for specific GPU architectures and CPU instruction sets using Flox and Nix. These builds match the optimizations of corresponding PyTorch variants from `../build-pytorch/`.

## Overview

This project provides **targeted TorchVision builds** that are optimized for specific hardware, resulting in:

- **Matched optimizations** - Pairs with PyTorch GPU/CPU configurations
- **Smaller binaries** - Only include code for your target GPU architecture
- **Better performance** - CPU code optimized for specific instruction sets
- **Consistent stack** - TorchVision and PyTorch built with same optimizations
- **Reproducible** - Managed with Flox/Nix for consistent builds

## Version Compatibility Strategy

For building older or specific versions of TorchVision that require compatible PyTorch versions:

### Nixpkgs Pinning Approach
When the default nixpkgs has incompatible versions (e.g., PyTorch 2.8.0 with TorchVision 0.24.1), we pin nixpkgs to a specific commit that contains compatible versions:

```nix
let
  # Pin nixpkgs to specific commit with compatible versions
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/COMMIT_HASH.tar.gz";
  }) {
    config = {
      allowUnfree = true;  # Required for CUDA packages
      cudaSupport = true;
    };
  };
in
  # Use nixpkgs_pinned.python3Packages instead of pkgs.python3Packages
```

**Current pinning**: Commit `fe5e41d7ffc0421f0913e8472ce6238ed0daf8e3` provides PyTorch 2.8.0 and TorchVision 0.23.0 compatibility.

To find compatible versions:
1. Search nixpkgs history for commits containing both packages at desired versions
2. Update all nix expressions to use the pinned nixpkgs
3. Test builds to verify compatibility

## Current Status

**✅ 44 variants on main branch**

| Architecture | Variants | Status |
|--------------|----------|--------|
| SM120 (RTX 5090) | 6/6 | ✅ Complete |
| CPU-only | 6/6 | ✅ Complete |
| SM100 (B100/B200) | 6/6 | ✅ Complete |
| SM90 (H100/L40S) | 6/6 | ✅ Complete |
| SM89 (RTX 4090) | 6/6 | ✅ Complete |
| SM86 (RTX 3090) | 6/6 | ✅ Complete |
| SM80 (A100/A30) | 6/6 | ✅ Complete |
| SM61 (GTX 1070/1080 Ti) | 1 | ✅ AVX variant (cuDNN, NNPACK disabled) |

**Total: 44 variants on main** (SM103 on `cuda-12_9`; SM110 + SM121 on `cuda-13_0`)

See **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** for detailed build matrix.

## Multi-Branch CUDA Strategy

Different GPU architectures require different minimum CUDA toolkit versions. Since the PyTorch override pattern binds `cudaPackages` from the nixpkgs scope, each CUDA version needs its own branch:

| Branch | CUDA Toolkit | Architectures | Variants |
|--------|-------------|---------------|----------|
| `main` | CUDA 12.8 | SM61, SM80–SM100, SM120, CPU | 44 |
| `cuda-12_9` | CUDA 12.9 | SM103 | 6 |
| `cuda-13_0` | CUDA 13.0 | SM110, SM121 | 12 |

**Why separate branches?**
- `sm_103` requires CUDA 12.9+ (not available in nixpkgs CUDA 12.8)
- `sm_110` requires CUDA 13.0+ (not available in nixpkgs CUDA 12.8 or 12.9)
- The PyTorch `override` pattern uses `cudaPackages` from the nixpkgs scope, so switching CUDA versions requires a different nixpkgs pin or overlay per branch

## Build Matrix

**49 variants** on main covering GPU architectures (SM61/SM80/SM86/SM89/SM90/SM100/SM120/SM121) × CPU instruction sets (AVX, AVX2, AVX-512, AVX-512 BF16, AVX-512 VNNI, ARMv8.2, ARMv9), plus CPU-only builds. SM103 and SM110 are on dedicated CUDA branches.

### Available Variants (49) ✅

#### SM121 (DGX Spark) - 6 variants ✅

| Package Name | CPU ISA | Platform | Primary Use Case |
|-------------|---------|----------|------------------|
| `torchvision-python313-cuda12_8-sm121-avx2` | AVX2 | x86_64-linux | DGX Spark + broad CPU compatibility |
| `torchvision-python313-cuda12_8-sm121-avx512` | AVX-512 | x86_64-linux | DGX Spark + general workloads |
| `torchvision-python313-cuda12_8-sm121-avx512bf16` | AVX-512 BF16 | x86_64-linux | DGX Spark + BF16 training |
| `torchvision-python313-cuda12_8-sm121-avx512vnni` | AVX-512 VNNI | x86_64-linux | DGX Spark + INT8 inference |
| `torchvision-python313-cuda12_8-sm121-armv8.2` | ARMv8.2-A | aarch64-linux | DGX Spark + ARM Graviton2 |
| `torchvision-python313-cuda12_8-sm121-armv9` | ARMv9-A | aarch64-linux | DGX Spark + ARM Grace |

#### SM120 (RTX 5090 / Blackwell) - 6 variants ✅

| Package Name | CPU ISA | Platform | Primary Use Case |
|-------------|---------|----------|------------------|
| `torchvision-python313-cuda12_8-sm120-avx2` | AVX2 | x86_64-linux | RTX 5090 + broad CPU compatibility |
| `torchvision-python313-cuda12_8-sm120-avx512` | AVX-512 | x86_64-linux | RTX 5090 + general workloads |
| `torchvision-python313-cuda12_8-sm120-avx512bf16` | AVX-512 BF16 | x86_64-linux | RTX 5090 + BF16 training |
| `torchvision-python313-cuda12_8-sm120-avx512vnni` | AVX-512 VNNI | x86_64-linux | RTX 5090 + INT8 inference |
| `torchvision-python313-cuda12_8-sm120-armv8.2` | ARMv8.2-A | aarch64-linux | RTX 5090 + ARM Graviton2 |
| `torchvision-python313-cuda12_8-sm120-armv9` | ARMv9-A | aarch64-linux | RTX 5090 + ARM Grace |

#### CPU-only - 6 variants ✅

| Package Name | CPU ISA | Platform | Primary Use Case |
|-------------|---------|----------|------------------|
| `torchvision-python313-cpu-avx2` | AVX2 | x86_64-linux | CPU-only + broad compatibility |
| `torchvision-python313-cpu-avx512` | AVX-512 | x86_64-linux | CPU-only + general workloads |
| `torchvision-python313-cpu-avx512bf16` | AVX-512 BF16 | x86_64-linux | CPU-only + BF16 training |
| `torchvision-python313-cpu-avx512vnni` | AVX-512 VNNI | x86_64-linux | CPU-only + INT8 inference |
| `torchvision-python313-cpu-armv8.2` | ARMv8.2-A | aarch64-linux | CPU-only + ARM Graviton2 |
| `torchvision-python313-cpu-armv9` | ARMv9-A | aarch64-linux | CPU-only + ARM Grace |

See **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** for the complete list of all 49 variants including:
- SM100 (Blackwell B100/B200) - 6 variants ✅
- SM90 (Hopper H100/L40S) - 6 variants ✅
- SM89 (Ada Lovelace RTX 4090/L40) - 6 variants ✅
- SM86 (Ampere RTX 3090/A40) - 6 variants ✅
- SM80 (Ampere Datacenter A100/A30) - 6 variants ✅
- SM61 (Pascal GTX 1070/1080 Ti) - 1 variant (AVX, NNPACK disabled) ✅

## GPU Architecture Reference

**SM121 (DGX Spark) - Compute Capability 12.1**
- Specialized Datacenter: DGX Spark
- Driver: NVIDIA 570+
- Pattern: Type A (uses `gpuArchSM = "sm_121"`)
- Features: Specialized datacenter workloads, high-performance computing

**SM120 (Blackwell) - Compute Capability 12.0**
- Consumer: RTX 5090
- Driver: NVIDIA 570+
- Pattern: Type B (uses `gpuArchNum = "12.0"`)
- Note: Requires PyTorch 2.7+ or nightly builds

**SM110 (Blackwell Thor/NVIDIA DRIVE) - Compute Capability 11.0** *(See `cuda-13_0` branch)*
- Automotive/Edge: NVIDIA DRIVE platforms (Thor, Orin+)
- Driver: NVIDIA 550+
- Pattern: Type A (uses `gpuArchSM = "sm_110"`)
- Features: Automotive AI, autonomous driving, edge computing
- **Requires CUDA 13.0+** — variants are on the `cuda-13_0` branch

**SM103 (Blackwell B300 Datacenter) - Compute Capability 10.3** *(See `cuda-12_9` branch)*
- Datacenter: B300
- Driver: NVIDIA 550+
- Pattern: Type A (uses `gpuArchSM = "sm_103"`)
- Features: Advanced Blackwell datacenter capabilities
- **Requires CUDA 12.9+** — variants are on the `cuda-12_9` branch

**SM100 (Blackwell Datacenter) - Compute Capability 10.0**
- Datacenter: B100, B200
- Driver: NVIDIA 550+
- Pattern: Type A (uses `gpuArchSM = "sm_100"`)
- Features: FP4 GEMV kernels, blockscaled datatypes, mixed input GEMM

**SM90 (Hopper) - Compute Capability 9.0**
- Datacenter: H100, H200, L40S
- Driver: NVIDIA 525+
- Pattern: Type A (uses `gpuArchSM = "sm_90"`)
- Features: Native FP8, Transformer Engine

**SM89 (Ada Lovelace) - Compute Capability 8.9**
- Consumer: RTX 4090, RTX 4080, RTX 4070 series
- Datacenter: L4, L40
- Driver: NVIDIA 520+
- Pattern: Type A (uses `gpuArchSM = "sm_89"`)
- Features: RT cores (3rd gen), Tensor cores (4th gen), DLSS 3

**SM86 (Ampere) - Compute Capability 8.6**
- Consumer: RTX 3090, RTX 3090 Ti, RTX 3080 Ti
- Datacenter: A5000, A40
- Driver: NVIDIA 470+
- Pattern: Type B (uses `gpuArchNum = "8.6"`)
- Features: RT cores, Tensor cores (2nd gen)

**SM80 (Ampere Datacenter) - Compute Capability 8.0**
- Datacenter: A100 (40GB/80GB), A30
- Driver: NVIDIA 450+
- Pattern: Type A (uses `gpuArchSM = "sm_80"`)
- Features: Multi-Instance GPU (MIG), Tensor cores (3rd gen), FP64 Tensor cores

**SM61 (Pascal) - Compute Capability 6.1**
- Consumer: GTX 1070, GTX 1080, GTX 1080 Ti
- Driver: NVIDIA 390+
- Pattern: Type C (uses `gpuArchSM = "6.1"` — dot notation required for older architectures)
- Features: CUDA cores, no Tensor cores
- **Note:** cuDNN 9.11+ dropped support for SM < 7.5; NNPACK requires AVX2+FMA3. Both are disabled for SM61-AVX builds.

## CPU Variant Guide

Choose the right CPU variant based on your hardware and workload:

**AVX (Maximum Compatibility)**
- Hardware: Intel Sandy Bridge+ (2011+), AMD Bulldozer+ (2011+)
- Use for: Oldest possible CPU support, legacy systems
- Choose when: Running on pre-2013 hardware or need absolute maximum compatibility

**AVX2 (Broad Compatibility)**
- Hardware: Intel Haswell+ (2013+), AMD Zen 1+ (2017+)
- Use for: Maximum compatibility, development, general workloads
- Choose when: Uncertain about CPU features or need portability

**AVX-512 (General Performance)**
- Hardware: Intel Skylake-X+ (2017+), AMD Zen 4+ (2022+)
- Use for: General FP32 training and inference on modern CPUs
- Choose when: You have AVX-512 CPU and need general-purpose performance

**AVX-512 BF16 (Mixed-Precision Training)**
- Hardware: Intel Cooper Lake+ (2020+), AMD Zen 4+ (2022+)
- Use for: BF16 (Brain Float 16) mixed-precision training
- Detection: `lscpu | grep bf16` or `/proc/cpuinfo` shows `avx512_bf16`

**AVX-512 VNNI (INT8 Inference)**
- Hardware: Intel Skylake-SP+ (2017+), AMD Zen 4+ (2022+)
- Use for: Quantized INT8 model inference acceleration
- Detection: `lscpu | grep vnni` or `/proc/cpuinfo` shows `avx512_vnni`

**ARMv8.2 (ARM Servers - Older)**
- Hardware: ARM Neoverse N1, Cortex-A75+, AWS Graviton2
- Use for: ARM servers without SVE2 support

**ARMv9 (ARM Servers - Modern)**
- Hardware: NVIDIA Grace, ARM Neoverse V1/V2, Cortex-X2+, AWS Graviton3+
- Use for: Modern ARM servers with SVE2 (Scalable Vector Extensions)
- Detection: `lscpu | grep sve` or `/proc/cpuinfo` shows `sve` and `sve2`

## Variant Selection Guide

### Quick Decision Tree

**1. Do you have an NVIDIA GPU?**
- NO → Use CPU-only variant (see BUILD_MATRIX.md)
- YES → Continue to step 2

**2. Which GPU do you have?**
```bash
# Check GPU model
nvidia-smi --query-gpu=name --format=csv,noheader

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

| Your GPU | Compute Cap | Use Architecture | Status |
|----------|-------------|------------------|--------|
| DGX Spark | 12.1 | **SM121** | ✅ Available |
| RTX 5090 | 12.0 | **SM120** | ✅ Available |
| NVIDIA DRIVE Thor, Orin+ | 11.0 | **SM110** | See `cuda-13_0` branch |
| B300 | 10.3 | **SM103** | See `cuda-12_9` branch |
| B100, B200 | 10.0 | **SM100** | ✅ Available |
| H100, H200, L40S | 9.0 | **SM90** | ✅ Available |
| RTX 4090, RTX 4080, RTX 4070 series, L4, L40 | 8.9 | **SM89** | ✅ Available |
| RTX 3090, RTX 3090 Ti, RTX 3080 Ti, A5000, A40 | 8.6 | **SM86** | ✅ Available |
| A100, A30 | 8.0 | **SM80** | ✅ Available |
| GTX 1070, GTX 1080, GTX 1080 Ti | 6.1 | **SM61** | ✅ Available (AVX only, cuDNN+NNPACK disabled) |

**3. Which CPU ISA should you use?**
```bash
# Check CPU features
lscpu | grep -E 'avx|sve'
# or
grep -E 'avx|sve' /proc/cpuinfo
```

| If you see... | Platform | Workload Type | Choose |
|--------------|----------|---------------|--------|
| `avx512_bf16` | x86-64 | BF16 training on CPU | `avx512bf16` |
| `avx512_vnni` | x86-64 | INT8 inference | `avx512vnni` |
| `avx512f` | x86-64 | General workloads | `avx512` |
| `avx2` (no avx512) | x86-64 | General workloads | `avx2` |
| `sve` and `sve2` | ARM | Modern ARM (Grace, Graviton3+) | `armv9` |
| Neither | ARM | Older ARM (Graviton2) | `armv8.2` |

## Quick Start

### Prerequisites

**IMPORTANT:** TorchVision requires a matching PyTorch variant from `../build-pytorch/`.

1. **Build matching PyTorch variant** first:
   ```bash
   cd ../build-pytorch
   flox activate
   flox build pytorch-python313-cuda12_8-sm120-avx512
   cd ../build-torchvision
   ```

2. **Flox** package manager installed
3. **NVIDIA GPU** (for CUDA builds) with appropriate drivers (570+ for SM120/SM121)

### Building a Variant

```bash
# Navigate to build-torchvision directory
cd build-torchvision

# Activate flox environment
flox activate

# Build a specific variant (49 variants on main)
flox build torchvision-python313-cuda12_8-sm120-avx512

# Result appears as symlink
ls -lh result-torchvision-python313-cuda12_8-sm120-avx512/

# Test the build
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python -c "
import torch, torchvision
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Example Scenarios

**Scenario 1: RTX 5090 + Intel i9 with AVX-512**
```bash
flox build torchvision-python313-cuda12_8-sm120-avx512
```

**Scenario 2: DGX Spark + AMD EPYC Zen 4**
```bash
flox build torchvision-python313-cuda12_8-sm121-avx512
```

**Scenario 3: RTX 5090 + Broad CPU compatibility**
```bash
flox build torchvision-python313-cuda12_8-sm120-avx2
```

**Scenario 4: RTX 5090 + ARM Grace Hopper**
```bash
flox build torchvision-python313-cuda12_8-sm120-armv9
```

## GPU Architecture Patterns (CRITICAL!)

TorchVision must match the GPU architecture pattern used by the corresponding PyTorch build. There are **THREE different patterns**:

### Pattern Type A: sm_XXX format (SM121, SM100, SM90, SM89, SM80)

**Used by:** SM121, SM100, SM90, SM89, SM80 (also SM110 and SM103 on their respective CUDA branches)

```nix
gpuArchNum = "121";        # For CMAKE_CUDA_ARCHITECTURES
gpuArchSM = "sm_121";      # For TORCH_CUDA_ARCH_LIST
gpuTargets = [ gpuArchSM ]; # Uses sm_121
```

### Pattern Type B: Decimal format without gpuArchSM (SM120, SM86)

**Used by:** SM120, SM86 only

```nix
# PyTorch's CMake accepts numeric format (12.0/8.6) not sm_XXX for these architectures
gpuArchNum = "12.0";       # Or "8.6" for SM86
# NO gpuArchSM variable
gpuTargets = [ gpuArchNum ]; # Uses numeric format directly
```

### Pattern Type C: Dot notation in gpuArchSM (SM61 and older architectures)

**Used by:** SM61 (and likely other pre-SM80 architectures)

```nix
# Older architectures require dot notation in TORCH_CUDA_ARCH_LIST
gpuArchNum = "61";
gpuArchSM = "6.1";         # Dot notation, NOT "sm_61"
gpuTargets = [ gpuArchSM ]; # Uses "6.1"
```

**Why:** PyTorch's CMake/TORCH_CUDA_ARCH_LIST expects dot notation (e.g., `"6.1"`, `"7.0"`, `"7.5"`) for older compute capabilities. The `sm_XXX` format (Type A) is used by newer architectures (SM80+).

**CRITICAL:** Always check the PyTorch pattern before creating TorchVision variants!

```bash
# Verify pattern for any architecture
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm{ARCH}-*.nix | head -5
```

## Documentation

This project includes comprehensive documentation:

- **[QUICKSTART.md](./QUICKSTART.md)** - Quick start guide with examples
- **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** - Complete build matrix (49 variants on main)
- **[RECIPE_TEMPLATE.md](./RECIPE_TEMPLATE.md)** - Templates for creating new variants
- **[TEST_GUIDE.md](./TEST_GUIDE.md)** - Testing procedures and examples
- **[README.md](./README.md)** - This file (overview and reference)

## Adding More Variants

All 49 variants on main are complete! To add support for future GPU architectures:

### Step 1: Check PyTorch Pattern

**CRITICAL:** Before creating TorchVision variants, check the PyTorch pattern!

```bash
# Example: Check SM90 pattern
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm90-*.nix | head -5
```

### Step 2: Use RECIPE_TEMPLATE.md

See **[RECIPE_TEMPLATE.md](./RECIPE_TEMPLATE.md)** for:
- Complete templates for Pattern Type A, Pattern Type B, and Pattern Type C
- Variable lookup tables
- Step-by-step instructions
- Examples for all architectures

### Step 3: Generate All 6 Variants

For each GPU architecture, create 6 variants:
- 4 x86-64: `avx2`, `avx512`, `avx512bf16`, `avx512vnni`
- 2 ARM: `armv8.2`, `armv9`

### Step 4: Test and Commit

```bash
# Build one variant to test
flox build torchvision-python313-cuda12_8-sm{ARCH}-avx512

# Test it (see TEST_GUIDE.md)
./result-torchvision-python313-cuda12_8-sm{ARCH}-avx512/bin/python [test script]

# Commit
git add .flox/pkgs/torchvision-python313-cuda12_8-sm{ARCH}-*.nix
git commit -m "Add SM{ARCH} TorchVision variants - all 6 CPU ISAs"
```

## Build Configuration Details

### GPU Builds

GPU-optimized builds use:
- **Custom PyTorch** with matching GPU/CPU configuration
- **CUDA 12.8** from cudaPackages
- **Targeted compilation** via `gpuTargets` parameter

Each GPU variant only compiles kernels for its specific SM architecture.

### TorchVision Build Pattern

TorchVision variants use a custom build pattern:

```nix
# Create custom PyTorch with matching configuration
customPytorch = (python3Packages.pytorch.override {
  cudaSupport = true;
  gpuTargets = [ gpuArchNum ];  # or gpuArchSM for SM121
}).overrideAttrs (oldAttrs: {
  ninjaFlags = [ "-j32" ];
  requiredSystemFeatures = [ "big-parallel" ];

  preConfigure = (oldAttrs.preConfigure or "") + ''
    export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export MAX_JOBS=32
  '';
});

# Build TorchVision with custom PyTorch
(python3Packages.torchvision.override {
  torch = customPytorch;
}).overrideAttrs (oldAttrs: {
  pname = "torchvision-python313-cuda12_8-sm120-avx512";
  ninjaFlags = [ "-j32" ];
  requiredSystemFeatures = [ "big-parallel" ];

  preConfigure = (oldAttrs.preConfigure or "") + ''
    export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export MAX_JOBS=32
  '';
})
```

### Memory Saturation Prevention (CRITICAL!)

**Problem:** TorchVision builds trigger multiple concurrent derivations (PyTorch + TorchVision + dependencies), each spawning unlimited CUDA compiler processes (`nvcc`, `cicc`, `ptxas`). On systems with `max-jobs = auto` and `cores = 0` (like Flox), this can saturate memory and spawn 95+ concurrent processes.

**Solution:** Use `requiredSystemFeatures = [ "big-parallel" ];` in **BOTH** the customPytorch and torchvision sections:

```nix
customPytorch = (python3Packages.pytorch.override {
  cudaSupport = true;
  gpuTargets = [ gpuArchNum ];
}).overrideAttrs (oldAttrs: {
  ninjaFlags = [ "-j32" ];
  requiredSystemFeatures = [ "big-parallel" ];  # ← CRITICAL!

  preConfigure = (oldAttrs.preConfigure or "") + ''
    export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export MAX_JOBS=32
  '';
});

in
  (python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda12_8-sm120-avx512";
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];  # ← CRITICAL!

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32
    '';
  })
```

**Why this works:**
- `requiredSystemFeatures = [ "big-parallel" ]` tells Nix daemon to serialize resource-heavy builds
- Prevents concurrent builds of PyTorch + TorchVision + dependencies
- Controls CUDA compiler parallelism at the Nix orchestration level
- `ninjaFlags = [ "-j32" ]` limits ninja build parallelism to 32 cores
- `MAX_JOBS=32` controls Python setuptools parallelism

**What doesn't work:**
- Environment variables like `NIX_BUILD_CORES` or `CMAKE_BUILD_PARALLEL_LEVEL` are ineffective
- CUDA compiler tools spawn their own processes outside ninja's control
- Only Nix-level serialization with `requiredSystemFeatures` prevents concurrent derivation builds

**All TorchVision variants in this repository include this fix.** If creating new variants, see RECIPE_TEMPLATE.md for the correct pattern.

## PyTorch Dependency Resolution (TODO)

**Current Status:** TorchVision variants use `pytorch.override` to create a custom PyTorch with matching GPU/CPU configuration.

**Future Work:** Set up proper dependency resolution to reference actual PyTorch packages from `../build-pytorch` instead of using `pytorch.override`. This will ensure exact matching between TorchVision and PyTorch builds.

## Publishing to Flox Catalog

Once builds are validated, publish them for team use:

```bash
# Ensure git remote is configured
git remote add origin <your-repo-url>
git push origin main

# Publish to your Flox organization
flox publish -o <your-org> torchvision-python313-cuda12_8-sm120-avx512
flox publish -o <your-org> torchvision-python313-cuda12_8-sm121-avx512

# Users install with:
flox install <your-org>/torchvision-python313-cuda12_8-sm120-avx512
```

**IMPORTANT:** Ensure matching PyTorch variants are also published!

## Build Times & Requirements

⚠️ **Note**: TorchVision builds are faster than PyTorch builds:

- **Time**: 30-90 minutes per variant (depends on CPU cores)
- **Disk**: ~8-10GB per build (source + build artifacts)
- **Memory**: 8GB+ RAM recommended
- **CPU**: Multi-core system recommended

**Recommendation**: Build on CI/CD runners and publish to your Flox catalog. Users then install pre-built packages instantly.

## Testing

See **[TEST_GUIDE.md](./TEST_GUIDE.md)** for comprehensive testing procedures.

Quick test:
```bash
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision
import torchvision.models as models

# Check versions
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test model loading and inference
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"✓ ResNet18 inference successful! Output shape: {output.shape}")
EOF
```

## Directory Structure

```
build-torchvision/
├── .flox/
│   ├── env/              # Flox environment configuration
│   │   └── manifest.toml
│   └── pkgs/             # Nix package definitions (49 variants on main)
│       ├── torchvision-python313-cuda12_8-sm121-*.nix  # 6 SM121 variants ✅
│       ├── torchvision-python313-cuda12_8-sm120-*.nix  # 6 SM120 variants ✅
│       ├── torchvision-python313-cuda12_8-sm100-*.nix  # 6 SM100 variants ✅
│       ├── torchvision-python313-cuda12_8-sm90-*.nix   # 6 SM90 variants ✅
│       ├── torchvision-python313-cuda12_8-sm89-*.nix   # 6 SM89 variants ✅
│       ├── torchvision-python313-cuda12_8-sm86-*.nix   # 6 SM86 variants ✅
│       ├── torchvision-python313-cuda12_8-sm80-*.nix   # 6 SM80 variants ✅
│       ├── torchvision-python313-cuda12_8-sm61-avx-no-nnpack.nix # 1 SM61 variant (Pascal, cuDNN+NNPACK disabled) ✅
│       └── torchvision-python313-cpu-*.nix             # 6 CPU-only variants ✅
├── .git/                 # Git repository
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
├── BUILD_MATRIX.md       # Complete build matrix
├── RECIPE_TEMPLATE.md    # Templates for creating variants
└── TEST_GUIDE.md         # Testing procedures
```

## Relationship to build-pytorch

This project is a companion to `../build-pytorch/` and follows the same:
- Architecture support matrix (8 GPU on main + 7 CPU ISAs; 2 GPU on CUDA branches)
- Naming conventions
- Build patterns
- Documentation structure

**Key difference:** TorchVision depends on matching PyTorch variants.

## Version Information

- **TorchVision**: 0.24.1 (from nixpkgs)
- **PyTorch**: 2.9.1 (from nixpkgs, compatible with TorchVision 0.24.1)
- **Python**: 3.13
- **CUDA**: 12.8
- **Platform**: Linux (x86_64-linux, aarch64-linux)

## Troubleshooting

### "Package not found"

**Problem:** TorchVision variant doesn't exist yet.

**Solution:** All 49 variants on main are complete. SM103 variants are on the `cuda-12_9` branch and SM110 variants are on the `cuda-13_0` branch. Check that you're using the correct package name format and branch. See BUILD_MATRIX.md for the full list of available variants.

### "PyTorch dependency not found"

**Problem:** Matching PyTorch variant doesn't exist.

**Solution:** Build the matching PyTorch variant first:
```bash
cd ../build-pytorch
flox build pytorch-python313-cuda12_8-sm120-avx512
cd ../build-torchvision
flox build torchvision-python313-cuda12_8-sm120-avx512
```

### "Wrong GPU pattern"

**Problem:** Created variant using wrong pattern (Type A vs Type B vs Type C).

**Solution:** Always check PyTorch pattern first:
```bash
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm{ARCH}-*.nix | head -5
```

## Related Documentation

- [PyTorch CUDA Architecture List](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
- [Flox Build Documentation](https://flox.dev/docs/reference/command-reference/flox-build/)
- [TorchVision Documentation](https://pytorch.org/vision/stable/index.html)
- **[../build-pytorch/](../build-pytorch/)** - PyTorch build environment (dependency)

## Contributing

This is a personal build system for creating optimized TorchVision variants. The structure follows the established patterns from `build-pytorch`.

To add new variants:
1. Check PyTorch pattern first (critical!)
2. Use RECIPE_TEMPLATE.md for templates
3. Test locally with `flox build <variant>`
4. Verify with TEST_GUIDE.md procedures
5. Commit and document changes

## License

This build environment configuration follows the same licensing as TorchVision and nixpkgs packages.

---

**Next Steps:**
1. ✅ 49 variants on main + 6 on cuda-12_9 + 12 on cuda-13_0 (67 total across branches)
2. Test variants on target hardware across all architectures
3. Set up proper PyTorch dependency resolution
4. Publish to FloxHub for team distribution
5. Document testing procedures and validation results
6. Create comprehensive performance benchmarks

See **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** for complete variant documentation.
