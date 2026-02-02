# Quick Start Guide

## Choosing Your Variant

**44 variants complete on main** branch (44/44 implemented - 100% ✅). SM121 (6 variants) is on the `cuda-13_0` branch. Choose based on your hardware to match your PyTorch variant.

> **Note:** SM103 and SM110 variants have been moved to CUDA-version-specific branches (`cuda-12_9` and `cuda-13_0` respectively). Check out the appropriate branch to build those variants.

**IMPORTANT:** TorchVision requires a matching PyTorch variant. Install PyTorch first from `../build-pytorch/`.

### Quick Selection

**Have NVIDIA GPU?**
```bash
# Check your GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

| Your GPU | Compute Cap | Architecture | Example Package | Status |
|----------|-------------|--------------|-----------------|--------|
| DGX Spark | 12.1 | SM121 | `torchvision-python313-cuda13_0-sm121-avx512` | ✅ Available (6 variants, on `cuda-13_0` branch) |
| RTX 5090 | 12.0 | SM120 | `torchvision-python313-cuda12_8-sm120-avx512` | ✅ Available (6 variants) |
| NVIDIA DRIVE Thor, Orin+ | 11.0 | SM110 | `torchvision-python313-cuda13_0-sm110-avx512` | See `cuda-13_0` branch |
| B300 | 10.3 | SM103 | `torchvision-python313-cuda12_9-sm103-avx512` | See `cuda-12_9` branch |
| B100, B200 | 10.0 | SM100 | `torchvision-python313-cuda12_8-sm100-avx512` | ✅ Available (6 variants) |
| H100, L40S | 9.0 | SM90 | `torchvision-python313-cuda12_8-sm90-avx512` | ✅ Available (6 variants) |
| RTX 4090, L40 | 8.9 | SM89 | `torchvision-python313-cuda12_8-sm89-avx512` | ✅ Available (6 variants) |
| RTX 3090, A40 | 8.6 | SM86 | `torchvision-python313-cuda12_8-sm86-avx512` | ✅ Available (6 variants) |
| A100, A30 | 8.0 | SM80 | `torchvision-python313-cuda12_8-sm80-avx512` | ✅ Available (6 variants) |

**CPU-only (no GPU)?**
```bash
# Check CPU features
lscpu | grep -E 'avx512|sve'
```

- See `avx512_bf16`? → Use `torchvision-python313-cpu-avx512bf16` ✅
- See `avx512_vnni`? → Use `torchvision-python313-cpu-avx512vnni` ✅
- See `avx512f`? → Use `torchvision-python313-cpu-avx512` ✅
- See `avx2` only? → Use `torchvision-python313-cpu-avx2` ✅
- See `sve2` (ARM)? → Use `torchvision-python313-cpu-armv9` ✅
- ARM without sve2? → Use `torchvision-python313-cpu-armv8.2` ✅

**Naming format:** `torchvision-python313-{cuda12_8-smXX|cpu}-{cpu-isa}`

## Current Status

### Available Variants (44/44 on main) ✅ COMPLETE

> **Note:** SM103 (6 variants) moved to `cuda-12_9` branch. SM110 + SM121 (12 variants) moved to `cuda-13_0` branch.

**SM121 (DGX Spark) - 6 variants** *(on `cuda-13_0` branch, not main):*
- `torchvision-python313-cuda13_0-sm121-avx2` (x86_64)
- `torchvision-python313-cuda13_0-sm121-avx512` (x86_64)
- `torchvision-python313-cuda13_0-sm121-avx512bf16` (x86_64)
- `torchvision-python313-cuda13_0-sm121-avx512vnni` (x86_64)
- `torchvision-python313-cuda13_0-sm121-armv8.2` (aarch64)
- `torchvision-python313-cuda13_0-sm121-armv9` (aarch64)

**SM120 (RTX 5090) - 6 variants:**
- `torchvision-python313-cuda12_8-sm120-avx2` (x86_64)
- `torchvision-python313-cuda12_8-sm120-avx512` (x86_64)
- `torchvision-python313-cuda12_8-sm120-avx512bf16` (x86_64)
- `torchvision-python313-cuda12_8-sm120-avx512vnni` (x86_64)
- `torchvision-python313-cuda12_8-sm120-armv8.2` (aarch64)
- `torchvision-python313-cuda12_8-sm120-armv9` (aarch64)

**Plus 36 additional variants** for SM100, SM90, SM89, SM86, SM80, and CPU-only builds.

See **BUILD_MATRIX.md** for the complete list of all 44 variants on `main`. SM103 is on `cuda-12_9`; SM110 and SM121 are on `cuda-13_0`.

## Building TorchVision Variants

### Prerequisites

**IMPORTANT:** TorchVision depends on PyTorch. Ensure matching PyTorch variant exists in `../build-pytorch/`:

```bash
# Check if matching PyTorch exists
ls ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm120-avx512.nix

# If not, build PyTorch first
cd ../build-pytorch
flox activate
flox build pytorch-python313-cuda12_8-sm120-avx512
cd ../build-torchvision
```

### 1. Enter the Environment

```bash
cd /home/daedalus/dev/builds/build-torchvision
flox activate
```

### 2. Build a Variant

```bash
# SM120 variants (RTX 5090) - Available now
flox build torchvision-python313-cuda12_8-sm120-avx2        # Broad compatibility
flox build torchvision-python313-cuda12_8-sm120-avx512      # General performance
flox build torchvision-python313-cuda12_8-sm120-avx512bf16  # BF16 training
flox build torchvision-python313-cuda12_8-sm120-avx512vnni  # INT8 inference

# SM121 variants (DGX Spark) - requires `git checkout cuda-13_0` first
flox build torchvision-python313-cuda13_0-sm121-avx512      # General performance

# ARM variants (if on ARM server)
flox build torchvision-python313-cuda12_8-sm120-armv9       # Grace, Graviton3+
flox build torchvision-python313-cuda12_8-sm120-armv8.2     # Graviton2

# Other GPU architectures - All available!
flox build torchvision-python313-cuda12_8-sm90-avx512       # H100 ✅
flox build torchvision-python313-cuda12_8-sm89-avx512       # RTX 4090 ✅
flox build torchvision-python313-cuda12_8-sm86-avx512       # RTX 3090 ✅
flox build torchvision-python313-cuda12_8-sm80-avx512       # A100 ✅

# CPU-only variants
flox build torchvision-python313-cpu-avx2                    # CPU only ✅
```

### 3. Use the Built Package

```bash
# Result appears as a symlink
ls -lh result-torchvision-python313-cuda12_8-sm120-avx512

# Test it
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python -c "import torchvision; print(torchvision.__version__)"

# Activate the environment
source result-torchvision-python313-cuda12_8-sm120-avx512/bin/activate
python -c "import torch, torchvision; print(f'PyTorch: {torch.__version__}, TorchVision: {torchvision.__version__}')"
```

## What Happened

The build command:
```bash
flox build torchvision-python313-cuda12_8-sm120-avx512
```

Is doing:
1. Reading `.flox/pkgs/torchvision-python313-cuda12_8-sm120-avx512.nix`
2. Creating a custom PyTorch with matching GPU/CPU configuration
3. Calling `python3Packages.torchvision.override` with the custom PyTorch
4. Setting `CXXFLAGS="-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"` for CPU optimization
5. Compiling TorchVision from source (~30-90 minutes)
6. Creating a result symlink: `./result-torchvision-python313-cuda12_8-sm120-avx512`

## Successful Output

You should see output like this:

```
Building python3.13-torchvision-python313-cuda12_8-sm120-avx512-0.21.0 in Nix expression mode
this derivation will be built:
  /nix/store/...-python3.13-torchvision-python313-cuda12_8-sm120-avx512-0.21.0.drv
these X paths will be fetched (...):
```

This means:
- ✅ Flox found your Nix expression
- ✅ The package name is correct
- ✅ Dependencies (including PyTorch) are being resolved
- ✅ The build will compile TorchVision with your custom flags

## Build Time Warning

**CPU-only builds:** 30-60 minutes on a multi-core system
**GPU builds:** 1-2 hours (CUDA compilation overhead + PyTorch compilation)

**Note:** TorchVision builds are faster than PyTorch builds because the package is smaller.

**Recommendation:** Start with one variant to validate everything works, then queue up multiple builds to run in parallel or overnight.

## Publishing (After Build Completes)

Once you have successfully built packages:

```bash
# Push to git remote
git remote add origin <your-repo-url>
git push origin main

# Publish to Flox catalog (requires flox auth login)
flox publish -o <your-org> torchvision-python313-cuda12_8-sm120-avx512
flox publish -o <your-org> torchvision-python313-cuda12_8-sm120-avx2
flox publish -o <your-org> torchvision-python313-cuda13_0-sm121-avx512  # on cuda-13_0 branch

# Users can then install with:
flox install <your-org>/torchvision-python313-cuda12_8-sm120-avx512
```

**IMPORTANT:** Ensure matching PyTorch variants are also published so users can install the complete stack!

## Adding More Variants

### Step 1: Check PyTorch Pattern

**CRITICAL:** Before creating TorchVision variants for a new GPU architecture, check the PyTorch pattern!

```bash
# Example: Check SM90 pattern
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm90-*.nix | head -5
```

**If you see `gpuArchSM`:** Use Pattern Type A (like SM121)
**If you see only `gpuArchNum`:** Use Pattern Type B (like SM120)

### Step 2: Use RECIPE_TEMPLATE.md

See **RECIPE_TEMPLATE.md** for complete templates and variable lookup tables.

### Step 3: Copy and Modify

#### Example: Creating SM90 variants (Pattern Type A)

```bash
# SM90 uses Pattern Type A (like SM121, which is on cuda-13_0 branch)
# Copy an existing Pattern Type A file as template (e.g., SM100 on main, or SM121 on cuda-13_0)
cp .flox/pkgs/torchvision-python313-cuda12_8-sm100-avx512.nix \
   .flox/pkgs/torchvision-python313-cuda12_8-sm90-avx512.nix
```

Edit the new file:
```nix
let
  # GPU target: SM90 (Hopper architecture - H100, L40S)
  gpuArchNum = "90";        # For CMAKE_CUDA_ARCHITECTURES (just the integer)
  gpuArchSM = "sm_90";      # For TORCH_CUDA_ARCH_LIST (with sm_ prefix)

  # CPU optimization: AVX-512
  cpuFlags = [
    "-mavx512f"    # AVX-512 Foundation
    "-mavx512dq"   # Doubleword and Quadword instructions
    "-mavx512vl"   # Vector Length extensions
    "-mavx512bw"   # Byte and Word instructions
    "-mfma"        # Fused multiply-add
  ];

  customPytorch = (python3Packages.pytorch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];  # Uses sm_90 format for SM90
  }).overrideAttrs (oldAttrs: {
    # CRITICAL: Prevent memory saturation during builds
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

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
    pname = "torchvision-python313-cuda12_8-sm90-avx512";  # Updated name

    # CRITICAL: Prevent memory saturation during builds
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: SM90 (Hopper: H100, L40S)"
      echo "CPU Features: AVX-512"
      echo "CUDA: 12.8 (Compute Capability 9.0)"
      echo "CXXFLAGS: $CXXFLAGS"
      echo "Build parallelism: 32 cores max"
      echo "========================================="
    '';
    # ... Update meta sections
  })
```

**Key changes:**
- Line 3 comment: Change to "SM90 (Hopper architecture - H100, L40S)"
- Line 5-6: Change `gpuArchNum = "90"` and `gpuArchSM = "sm_90"`
- Line 37: Change `pname = "torchvision-python313-cuda12_8-sm90-avx512"`
- Update echo statements and meta description to reference H100/L40S

**CRITICAL:** Always include these in BOTH customPytorch and torchvision sections:
- `ninjaFlags = [ "-j32" ];` - Limits ninja build parallelism
- `requiredSystemFeatures = [ "big-parallel" ];` - Prevents memory saturation
- `export MAX_JOBS=32` - Limits Python setuptools parallelism

See README.md "Memory Saturation Prevention" section for technical details.

### Step 4: Create All 6 Variants for the Architecture

For each GPU architecture, create 6 variants:
- 4 x86-64: `avx2`, `avx512`, `avx512bf16`, `avx512vnni`
- 2 ARM: `armv8.2`, `armv9`

Use **RECIPE_TEMPLATE.md** for the exact templates and variable substitutions.

### Step 5: Commit and Build

```bash
git add .flox/pkgs/torchvision-python313-cuda12_8-sm90-*.nix
git commit -m "Add SM90 (H100/L40S) TorchVision variants"
flox build torchvision-python313-cuda12_8-sm90-avx512
```

## Pattern Type Reference

### Pattern Type A (SM121, SM110, SM103, SM100, SM90, SM89, SM80)

```nix
gpuArchNum = "121";        # For CMAKE_CUDA_ARCHITECTURES
gpuArchSM = "sm_121";      # For TORCH_CUDA_ARCH_LIST
gpuTargets = [ gpuArchSM ]; # Uses sm_121
```

### Pattern Type B (SM120, SM86)

```nix
# PyTorch's CMake accepts numeric format (12.0/8.6) not sm_XXX for these architectures
gpuArchNum = "12.0";       # Or "8.6" for SM86
# NO gpuArchSM variable
gpuTargets = [ gpuArchNum ]; # Uses numeric format directly
```

**ALWAYS check PyTorch pattern before creating variants!**

## Example: Creating SM90 Variants (Full Workflow)

```bash
# 1. Verify PyTorch SM90 pattern
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm90-*.nix | head -5

# Expected output shows Pattern Type A (sm_XXX format)
# gpuArchNum = "90";
# gpuArchSM = "sm_90";
# gpuTargets = [ gpuArchSM ];

# 2. Use RECIPE_TEMPLATE.md to create all 6 SM90 variants
# (Copy Pattern Type A template, substitute variables accordingly)

# 3. Verify files created
ls .flox/pkgs/torchvision-python313-cuda12_8-sm90-*.nix

# Should see:
# torchvision-python313-cuda12_8-sm90-avx2.nix
# torchvision-python313-cuda12_8-sm90-avx512.nix
# torchvision-python313-cuda12_8-sm90-avx512bf16.nix
# torchvision-python313-cuda12_8-sm90-avx512vnni.nix
# torchvision-python313-cuda12_8-sm90-armv8.2.nix
# torchvision-python313-cuda12_8-sm90-armv9.nix

# 4. Commit
git add .flox/pkgs/torchvision-python313-cuda12_8-sm90-*.nix
git commit -m "Add SM90 (H100/L40S) TorchVision variants - all 6 CPU ISAs"

# 5. Build one variant to test
flox build torchvision-python313-cuda12_8-sm90-avx512

# 6. If successful, build the rest
flox build torchvision-python313-cuda12_8-sm90-avx2
flox build torchvision-python313-cuda12_8-sm90-avx512bf16
flox build torchvision-python313-cuda12_8-sm90-avx512vnni
flox build torchvision-python313-cuda12_8-sm90-armv8.2
flox build torchvision-python313-cuda12_8-sm90-armv9
```

## Testing Your Build

See **TEST_GUIDE.md** (when created) for comprehensive testing procedures.

Quick test:
```bash
# Test import and version
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python -c "
import torch
import torchvision
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test GPU detection (GPU builds only)
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
"
```

## Next Steps

See the full documentation:
- **RECIPE_TEMPLATE.md** - Templates for creating any variant
- **BUILD_MATRIX.md** - Complete build matrix and variant status
- **TEST_GUIDE.md** - Testing procedures (TO CREATE)
- **README.md** - Complete guide and overview

## Common Issues

### Issue: "Package not found"

**Problem:** TorchVision variant doesn't exist yet.

**Solution:** All 44 variants on `main` are complete. For SM103, check out `cuda-12_9`; for SM110 or SM121, check out `cuda-13_0`. See `BUILD_MATRIX.md` for the full list of available variants.

### Issue: "PyTorch dependency not found"

**Problem:** Matching PyTorch variant doesn't exist in `../build-pytorch/`.

**Solution:** Build the matching PyTorch variant first:
```bash
cd ../build-pytorch
flox build pytorch-python313-cuda12_8-sm120-avx512
cd ../build-torchvision
flox build torchvision-python313-cuda12_8-sm120-avx512
```

### Issue: "Wrong GPU pattern"

**Problem:** Created variant using wrong pattern (Type A vs Type B).

**Solution:** Always check PyTorch pattern first:
```bash
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm{ARCH}-*.nix | head -5
```

Then use the matching pattern for TorchVision.

## Quick Reference Card

| Architecture | GPU | Pattern | Status | Example Package |
|--------------|-----|---------|--------|-----------------|
| SM121 | DGX Spark | Type A (sm_121) | ✅ Available (cuda-13_0 branch) | `torchvision-python313-cuda13_0-sm121-avx512` |
| SM120 | RTX 5090 | Type B (12.0) | ✅ Available | `torchvision-python313-cuda12_8-sm120-avx512` |
| SM110 | DRIVE Thor | Type A (sm_110) | See `cuda-13_0` branch | `torchvision-python313-cuda13_0-sm110-avx512` |
| SM103 | B300 | Type A (sm_103) | See `cuda-12_9` branch | `torchvision-python313-cuda12_9-sm103-avx512` |
| SM100 | B100/B200 | Type A (sm_100) | ✅ Available | `torchvision-python313-cuda12_8-sm100-avx512` |
| SM90 | H100/L40S | Type A (sm_90) | ✅ Available | `torchvision-python313-cuda12_8-sm90-avx512` |
| SM89 | RTX 4090 | Type A (sm_89) | ✅ Available | `torchvision-python313-cuda12_8-sm89-avx512` |
| SM86 | RTX 3090 | Type B (8.6) | ✅ Available | `torchvision-python313-cuda12_8-sm86-avx512` |
| SM80 | A100/A30 | Type A (sm_80) | ✅ Available | `torchvision-python313-cuda12_8-sm80-avx512` |
| CPU | None | N/A | ✅ Available | `torchvision-python313-cpu-avx512` |

**Pattern Type A (7 architectures):** `gpuArchSM = "sm_XXX"`, `gpuTargets = [ gpuArchSM ]`
**Pattern Type B (2 architectures):** `gpuArchNum = "X.X"`, `gpuTargets = [ gpuArchNum ]` (NO gpuArchSM)
