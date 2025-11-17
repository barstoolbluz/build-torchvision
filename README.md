# TorchVision Architecture-Specific Builds

Build custom TorchVision variants optimized for specific GPU architectures and CPU instruction sets using Flox and Nix. These builds match the optimizations of corresponding PyTorch variants from `../build-pytorch/`.

## Overview

This project provides **targeted TorchVision builds** that are optimized for specific hardware, resulting in:

- **Matched optimizations** - Pairs with PyTorch GPU/CPU configurations
- **Smaller binaries** - Only include code for your target GPU architecture
- **Better performance** - CPU code optimized for specific instruction sets
- **Consistent stack** - TorchVision and PyTorch built with same optimizations
- **Reproducible** - Managed with Flox/Nix for consistent builds

## Current Status

**✅ 12/60 variants implemented (20%)**

| Architecture | Variants | Status |
|--------------|----------|--------|
| SM121 (DGX Spark) | 6/6 | ✅ Complete |
| SM120 (RTX 5090) | 6/6 | ✅ Complete |
| SM110 (DRIVE Thor) | 0/6 | ❌ Not created |
| SM103 (B300) | 0/6 | ❌ Not created |
| SM100 (B100/B200) | 0/6 | ❌ Not created |
| SM90 (H100/L40S) | 0/6 | ❌ Not created |
| SM89 (RTX 4090) | 0/6 | ❌ Not created |
| SM86 (RTX 3090) | 0/6 | ❌ Not created |
| SM80 (A100/A30) | 0/6 | ❌ Not created |
| CPU-only | 0/6 | ❌ Not created |

**Total: 12/60 variants created**

See **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** for detailed build matrix and remaining variants.

## Build Matrix

**60 planned variants** covering GPU architectures (SM80/SM86/SM89/SM90/SM100/SM103/SM110/SM120/SM121) × CPU instruction sets (AVX2, AVX-512, AVX-512 BF16, AVX-512 VNNI, ARMv8.2, ARMv9), plus CPU-only builds.

### Available Variants (12/60)

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

### Remaining Variants (48/60)

See **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** for the complete list of remaining variants for:
- SM110 (NVIDIA DRIVE Thor, Orin+)
- SM103 (Blackwell B300)
- SM100 (Blackwell B100/B200)
- SM90 (Hopper H100/L40S)
- SM89 (Ada Lovelace RTX 4090/L40)
- SM86 (Ampere RTX 3090/A40)
- SM80 (Ampere Datacenter A100/A30)
- CPU-only (6 variants)

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

**SM110 (Blackwell Thor/NVIDIA DRIVE) - Compute Capability 11.0**
- Automotive/Edge: NVIDIA DRIVE platforms (Thor, Orin+)
- Driver: NVIDIA 550+
- Pattern: Type B (uses `gpuArchNum = "11.0"`)
- Features: Automotive AI, autonomous driving, edge computing

**SM103 (Blackwell B300 Datacenter) - Compute Capability 10.3**
- Datacenter: B300
- Driver: NVIDIA 550+
- Pattern: Type B (uses `gpuArchNum = "10.3"`)
- Features: Advanced Blackwell datacenter capabilities

**SM100 (Blackwell Datacenter) - Compute Capability 10.0**
- Datacenter: B100, B200
- Driver: NVIDIA 550+
- Pattern: Type B (uses `gpuArchNum = "10.0"`)
- Features: FP4 GEMV kernels, blockscaled datatypes, mixed input GEMM

**SM90 (Hopper) - Compute Capability 9.0**
- Datacenter: H100, H200, L40S
- Driver: NVIDIA 525+
- Pattern: Type B (uses `gpuArchNum = "9.0"`)
- Features: Native FP8, Transformer Engine

**SM89 (Ada Lovelace) - Compute Capability 8.9**
- Consumer: RTX 4090, RTX 4080, RTX 4070 series
- Datacenter: L4, L40
- Driver: NVIDIA 520+
- Pattern: Type B (uses `gpuArchNum = "8.9"`)
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
- Pattern: Type B (uses `gpuArchNum = "8.0"`)
- Features: Multi-Instance GPU (MIG), Tensor cores (3rd gen), FP64 Tensor cores

## CPU Variant Guide

Choose the right CPU variant based on your hardware and workload:

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
- NO → Use CPU-only variant (TO CREATE - see BUILD_MATRIX.md)
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
| NVIDIA DRIVE Thor, Orin+ | 11.0 | **SM110** | ❌ To create |
| B300 | 10.3 | **SM103** | ❌ To create |
| B100, B200 | 10.0 | **SM100** | ❌ To create |
| H100, H200, L40S | 9.0 | **SM90** | ❌ To create |
| RTX 4090, RTX 4080, RTX 4070 series, L4, L40 | 8.9 | **SM89** | ❌ To create |
| RTX 3090, RTX 3090 Ti, RTX 3080 Ti, A5000, A40 | 8.6 | **SM86** | ❌ To create |
| A100, A30 | 8.0 | **SM80** | ❌ To create |

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

# Build a specific variant (SM120 or SM121 only - others not created yet)
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

TorchVision must match the GPU architecture pattern used by the corresponding PyTorch build. There are **TWO different patterns**:

### Pattern Type A: sm_XXX format (SM121)

**Used by:** SM121 (and potentially future architectures)

```nix
gpuArchNum = "121";        # For CMAKE_CUDA_ARCHITECTURES
gpuArchSM = "sm_121";      # For TORCH_CUDA_ARCH_LIST
gpuTargets = [ gpuArchSM ]; # Uses sm_121
```

### Pattern Type B: Decimal format (SM120 and older)

**Used by:** SM120, SM110, SM103, SM100, SM90, SM89, SM86, SM80

```nix
# PyTorch's CMake accepts numeric format (12.0/9.0/8.9/etc) not sm_XXX
gpuArchNum = "12.0";       # Or "11.0", "10.3", "10.0", "9.0", "8.9", "8.6", "8.0"
# NO gpuArchSM variable
gpuTargets = [ gpuArchNum ]; # Uses numeric format directly
```

**CRITICAL:** Always check the PyTorch pattern before creating TorchVision variants!

```bash
# Verify pattern for any architecture
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm{ARCH}-*.nix | head -5
```

## Documentation

This project includes comprehensive documentation:

- **[QUICKSTART.md](./QUICKSTART.md)** - Quick start guide with examples
- **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** - Complete build matrix (12/60 variants tracked)
- **[RECIPE_TEMPLATE.md](./RECIPE_TEMPLATE.md)** - Templates for creating new variants
- **[TEST_GUIDE.md](./TEST_GUIDE.md)** - Testing procedures and examples
- **[README.md](./README.md)** - This file (overview and reference)

## Adding More Variants

To create variants for other GPU architectures (SM90, SM89, SM86, etc.):

### Step 1: Check PyTorch Pattern

**CRITICAL:** Before creating TorchVision variants, check the PyTorch pattern!

```bash
# Example: Check SM90 pattern
grep -E "gpuArchNum|gpuArchSM|gpuTargets" \
  ../build-pytorch/.flox/pkgs/pytorch-python313-cuda12_8-sm90-*.nix | head -5
```

### Step 2: Use RECIPE_TEMPLATE.md

See **[RECIPE_TEMPLATE.md](./RECIPE_TEMPLATE.md)** for:
- Complete templates for Pattern Type A and Pattern Type B
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
flox build torchvision-python313-cuda12_8-sm90-avx512

# Test it (see TEST_GUIDE.md)
./result-torchvision-python313-cuda12_8-sm90-avx512/bin/python [test script]

# Commit
git add .flox/pkgs/torchvision-python313-cuda12_8-sm90-*.nix
git commit -m "Add SM90 (H100/L40S) TorchVision variants - all 6 CPU ISAs"
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
  preConfigure = (oldAttrs.preConfigure or "") + ''
    export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
  '';
});

# Build TorchVision with custom PyTorch
(python3Packages.torchvision.override {
  torch = customPytorch;
}).overrideAttrs (oldAttrs: {
  pname = "torchvision-python313-cuda12_8-sm120-avx512";
  # ... additional configuration
})
```

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
│   └── pkgs/             # Nix package definitions
│       ├── torchvision-python313-cuda12_8-sm121-*.nix  # 6 SM121 variants ✅
│       ├── torchvision-python313-cuda12_8-sm120-*.nix  # 6 SM120 variants ✅
│       └── (48 more variants to create)
├── .git/                 # Git repository
├── README.md             # This file
├── QUICKSTART.md         # Quick start guide
├── BUILD_MATRIX.md       # Complete build matrix
├── RECIPE_TEMPLATE.md    # Templates for creating variants
└── TEST_GUIDE.md         # Testing procedures
```

## Relationship to build-pytorch

This project is a companion to `../build-pytorch/` and follows the same:
- Architecture support matrix (9 GPU + 6 CPU ISAs)
- Naming conventions
- Build patterns
- Documentation structure

**Key difference:** TorchVision depends on matching PyTorch variants.

## Version Information

- **TorchVision**: Latest from nixpkgs (tracks PyTorch version)
- **Python**: 3.13
- **CUDA**: 12.8
- **Platform**: Linux (x86_64-linux, aarch64-linux)
- **PyTorch compatibility**: TorchVision 0.21+ requires PyTorch 2.7+

## Troubleshooting

### "Package not found"

**Problem:** TorchVision variant doesn't exist yet.

**Solution:** Check BUILD_MATRIX.md for current status. Only SM121 and SM120 variants are currently created. Use RECIPE_TEMPLATE.md to create missing variants.

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

**Problem:** Created variant using wrong pattern (Type A vs Type B).

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
1. Create SM90 variants (H100/L40S) - High priority for datacenter users
2. Create SM89 variants (RTX 4090) - High priority for gaming/workstation users
3. Create SM86 variants (RTX 3090) - Mainstream workstation users
4. Create CPU-only variants - Development and testing
5. Create remaining datacenter variants (SM80, SM100, SM103, SM110)
6. Implement proper PyTorch dependency resolution

See **[BUILD_MATRIX.md](./BUILD_MATRIX.md)** for complete variant tracking and priorities.
