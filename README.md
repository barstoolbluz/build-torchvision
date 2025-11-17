# TorchVision Architecture-Specific Builds

Build custom TorchVision variants optimized for specific GPU architectures and CPU instruction sets using Flox and Nix.

## Overview

This project provides architecture-specific TorchVision builds that match the GPU/CPU optimizations of corresponding PyTorch variants from `build-pytorch`. Each TorchVision variant is compiled with targeted optimizations for specific hardware.

## Key Features

- **GPU-Optimized**: Builds for NVIDIA architectures (SM80-SM121)
- **CPU-Optimized**: AVX-512, AVX-512 BF16, AVX-512 VNNI, AVX2, ARMv8.2, ARMv9
- **Architecture-Specific**: Each build targets specific compute capabilities
- **Reproducible**: Managed with Flox/Nix for consistent builds

## Architecture Support

### GPU Architectures (CUDA 12.8)
- **SM121** (12.1) - DGX Spark (Specialized Datacenter)
- **SM120** (12.0) - Blackwell (RTX 5090)
- **SM110** (11.0) - NVIDIA DRIVE Thor, Orin+ (Automotive)
- **SM103** (10.3) - Blackwell B300 (Datacenter)
- **SM100** (10.0) - Blackwell B100/B200 (Datacenter)
- **SM90** (9.0) - Hopper (H100, L40S)
- **SM89** (8.9) - Ada Lovelace (RTX 4090, L40)
- **SM86** (8.6) - Ampere (RTX 3090, A40, A5000)
- **SM80** (8.0) - Ampere Datacenter (A100, A30)

### CPU Instruction Sets
- **AVX-512 BF16**: BF16 training on Intel Sapphire Rapids+
- **AVX-512 VNNI**: INT8 inference on Intel Cascade Lake+
- **AVX-512**: Intel Skylake-X+, AMD Zen 4+
- **AVX2**: Broad x86-64 compatibility
- **ARMv9**: AWS Graviton3+, Grace Hopper
- **ARMv8.2**: AWS Graviton2, NVIDIA Tegra

## Quick Start

### Prerequisites

1. **Matching PyTorch variant** from `build-pytorch`
2. **Flox** package manager
3. **NVIDIA GPU** (for CUDA builds) with appropriate drivers

### Building a Variant

```bash
# Navigate to build-torchvision directory
cd build-torchvision

# Activate flox environment
flox activate

# Build a specific variant
flox build torchvision-python313-cuda12_8-sm120-avx512
```

## Current Status

### Implemented
- ✅ Flox environment initialized
- ✅ Git repository initialized
- ✅ Sample .nix file (SM120+AVX512)
- ✅ Directory structure created

### TODO
- ⏳ Generate all 54 CUDA variants (9 GPU archs × 6 CPU ISAs)
- ⏳ Generate 6 CPU-only variants
- ⏳ Create build matrix documentation
- ⏳ Add test scripts
- ⏳ Create RECIPE_TEMPLATE.md for variant generation
- ⏳ Configure proper PyTorch dependency resolution

## Package Naming Convention

```
torchvision-python{VERSION}-cuda{CUDA_VERSION}-{GPU_ARCH}-{CPU_ISA}
```

Examples:
- `torchvision-python313-cuda12_8-sm120-avx512` - RTX 5090 + AVX-512
- `torchvision-python313-cuda12_8-sm90-avx512bf16` - H100 + AVX-512 BF16
- `torchvision-python313-cuda12_8-sm86-avx2` - RTX 3090 + AVX2

## Dependencies

### PyTorch Integration

TorchVision variants **must** be built against matching PyTorch variants:

```
torchvision-python313-cuda12_8-sm120-avx512
  ↓ depends on
pytorch-python313-cuda12_8-sm120-avx512
```

**Important**: GPU architecture and CPU ISA must match between PyTorch and TorchVision.

### Current Limitation

The sample .nix files currently reference nixpkgs' base `python3Packages.pytorch`. For production use, these should reference the actual PyTorch variants from `build-pytorch`.

**TODO**: Implement proper cross-project dependency resolution (options):
1. Reference `build-pytorch` packages via Flox
2. Use Nix overlays
3. Publish PyTorch to FloxHub and reference from there

## Directory Structure

```
build-torchvision/
├── .flox/
│   ├── env/              # Flox environment configuration
│   ├── pkgs/             # Nix package definitions (.nix files)
│   └── ...
├── .git/                 # Git repository
├── README.md             # This file
└── (TODO: additional docs and scripts)
```

## Building Process

### Build Flow

1. TorchVision .nix file specifies GPU/CPU configuration
2. Dependencies include matching PyTorch variant
3. Nix builds TorchVision with:
   - GPU architecture targeting (inherited from PyTorch)
   - CPU optimization flags (CXXFLAGS/CFLAGS)
   - CUDA 12.8 support (via cudaPackages)

### Sample Build Command

```bash
# Build single variant
flox build torchvision-python313-cuda12_8-sm120-avx512

# View build outputs
ls -l result*
```

## Version Information

- **TorchVision**: Latest from nixpkgs (tracks PyTorch version)
- **Python**: 3.13
- **CUDA**: 12.8
- **Platform**: Linux (x86_64-linux, aarch64-linux)

## Relationship to build-pytorch

This project is a companion to `build-pytorch` and follows the same:
- Architecture support matrix
- Naming conventions
- Build patterns
- Documentation structure

## Next Steps

1. **Generate all variants**: Create .nix files for all 60 combinations
2. **PyTorch dependency**: Implement proper dependency on `build-pytorch` packages
3. **Documentation**: Create QUICKSTART.md, RECIPE_TEMPLATE.md, TEST_GUIDE.md
4. **Testing**: Add test scripts for verifying builds
5. **CI/CD**: Add automated builds for all variants

## Contributing

This is a personal build system for creating optimized TorchVision variants. The structure follows the established patterns from `build-pytorch`.

## Notes

- Builds are architecture-specific and won't work on mismatched hardware
- CUDA builds require NVIDIA drivers (570+ for Blackwell)
- Build times vary (30-60 minutes typical for TorchVision)
- TorchVision must match PyTorch version for ABI compatibility

## License

Follows the licensing of TorchVision and nixpkgs packages.
