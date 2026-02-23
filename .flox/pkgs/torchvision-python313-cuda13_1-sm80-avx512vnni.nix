# TorchVision 0.24+ with PyTorch 2.10.0 for NVIDIA Ampere Datacenter (SM80: A100/A30) + AVX-512 VNNI
# Package name: torchvision-python313-cuda13_1-sm80-avx512vnni
#
# NOTE: This uses the same PyTorch 2.10.0 overlay approach as build-pytorch.
# See build-pytorch/docs/pytorch-2.10-cuda13-build-notes.md for details on fixes.
#
# MAGMA: Uses nixpkgs CUDA 13.1 with built-in compatibility fix.
# Reference: https://github.com/icl-utk-edu/magma/issues/61

{ pkgs ? import <nixpkgs> {} }:

let
  # Import nixpkgs at a specific revision with CUDA 13.1
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/2017d6d515f8a7b289fe06d3a880a7ec588c3900.tar.gz";
  }) {
    config = {
      allowUnfree = true;
      allowBroken = true;
      cudaSupport = true;
    };
    overlays = [
      # Overlay 1: Use CUDA 13.1
      (final: prev: { cudaPackages = final.cudaPackages_13_1; })

      # Overlay 2: Upgrade PyTorch to 2.10.0
      (final: prev: {
        python3Packages = prev.python3Packages.override {
          overrides = pfinal: pprev: {
            torch = pprev.torch.overrideAttrs (oldAttrs: rec {
              version = "2.10.0";

              src = prev.fetchFromGitHub {
                owner = "pytorch";
                repo = "pytorch";
                rev = "v${version}";
                hash = "sha256-RKiZLHBCneMtZKRgTEuW1K7+Jpi+tx11BMXuS1jC1xQ=";
                fetchSubmodules = true;
              };

              # Clear patches - nixpkgs patches are for 2.9.1 and won't apply to 2.10.0
              patches = [];
            });
          };
        };
      })
    ];
  };

  # GPU target: SM80 (Ampere Datacenter - A100/A30)
  gpuArchSM = "8.0";

  # CPU optimization: AVX-512 VNNI
  cpuFlags = [
    "-mavx512f"      # AVX-512 Foundation
    "-mavx512dq"     # Doubleword and Quadword instructions
    "-mavx512vl"     # Vector Length extensions
    "-mavx512bw"     # Byte and Word instructions
    "-mavx512vnni"   # Vector Neural Network Instructions
    "-mfma"          # Fused multiply-add
  ];

  # Custom PyTorch 2.10.0 with all CUDA 13.1 fixes
  customPytorch = (nixpkgs_pinned.python3Packages.torch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];
  }).overrideAttrs (oldAttrs: {
    pname = "pytorch210-for-torchvision-sm80-avx512vnni";

    # Clear patches
    patches = [];

    # Limit build parallelism
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    # CMake flags for CUDA 13.1 compatibility
    cmakeFlags = (oldAttrs.cmakeFlags or []) ++ [
      "-DTORCH_BUILD_VERSION=2.10.0"
      "-DCMAKE_CUDA_FLAGS=-I/build/cccl-compat"
      "-DCUDA_VERSION=13.1"
    ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32

      # Version fixes for PyTorch 2.10.0
      export PYTORCH_BUILD_VERSION=2.10.0
      echo "2.10.0" > version.txt

      # CCCL header path compatibility for CUTLASS
      mkdir -p /build/cccl-compat/cccl
      ln -sf ${nixpkgs_pinned.cudaPackages.cuda_cccl}/include/cuda /build/cccl-compat/cccl/cuda
      ln -sf ${nixpkgs_pinned.cudaPackages.cuda_cccl}/include/cub /build/cccl-compat/cccl/cub
      ln -sf ${nixpkgs_pinned.cudaPackages.cuda_cccl}/include/thrust /build/cccl-compat/cccl/thrust
      ln -sf ${nixpkgs_pinned.cudaPackages.cuda_cccl}/include/nv /build/cccl-compat/cccl/nv
      export CXXFLAGS="-I/build/cccl-compat $CXXFLAGS"
      export CFLAGS="-I/build/cccl-compat $CFLAGS"
      export CUDAFLAGS="-I/build/cccl-compat $CUDAFLAGS"
    '';

    # FindCUDAToolkit.cmake delegating stub
    postPatch = (oldAttrs.postPatch or "") + ''
      mkdir -p cmake/Modules
      cat > cmake/Modules/FindCUDAToolkit.cmake << 'EOF'
# Delegating stub for FindCUDAToolkit
if(NOT CUDAToolkit_FOUND)
  set(_orig_module_path "''${CMAKE_MODULE_PATH}")
  list(FILTER CMAKE_MODULE_PATH EXCLUDE REGEX "cmake/Modules")
  include(FindCUDAToolkit)
  set(CMAKE_MODULE_PATH "''${_orig_module_path}")
endif()
EOF
    '';
  });

in
  (nixpkgs_pinned.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda13_1-sm80-avx512vnni";

    # Propagate pytorch's out output for transitive torch availability
    propagatedBuildInputs = (oldAttrs.propagatedBuildInputs or []) ++ [ customPytorch.out ];

    # Limit build parallelism
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: ${gpuArchSM} (Ampere DC: A100)"
      echo "CPU Features: AVX-512 VNNI"
      echo "CUDA: 13.1"
      echo "PyTorch: 2.10.0 (with CUDA 13.1 fixes)"
      echo "MAGMA: Enabled (nixpkgs built-in fix)"
      echo "TorchVision: ${oldAttrs.version}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision for NVIDIA A100/A30 (SM80, Ampere DC) + AVX-512 VNNI with PyTorch 2.10.0";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA Ampere Datacenter architecture (SM80) - A100/A30
        - CPU: x86-64 with AVX-512 VNNI instruction set
        - CUDA: 13.1 with compute capability 8.0
        - PyTorch: 2.10.0 (with all CUDA 13.1 compatibility fixes)
        - MAGMA: Enabled (nixpkgs includes fix)
        - Python: 3.13

        Hardware requirements:
        - GPU: A100, A30, or other SM80 GPUs
        - CPU: Intel Skylake-SP+ (2017+), AMD Zen 4+ (2022+)
        - Driver: NVIDIA 450+ required
      '';
      platforms = [ "x86_64-linux" ];
    };
  })
