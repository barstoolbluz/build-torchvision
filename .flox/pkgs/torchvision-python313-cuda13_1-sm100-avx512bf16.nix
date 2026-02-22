# TorchVision 0.24+ with PyTorch 2.10.0 for NVIDIA Blackwell Datacenter (SM100: B100/B200) + AVX-512 BF16
# Package name: torchvision-python313-cuda13_1-sm100-avx512bf16
#
# NOTE: This uses the same PyTorch 2.10.0 overlay approach as build-pytorch.
# See build-pytorch/docs/pytorch-2.10-cuda13-build-notes.md for details on fixes.
#
# MAGMA is enabled via a CUDA 13.1 compatibility patch.
# Patch reference: https://github.com/icl-utk-edu/magma/issues/61

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

      # Overlay 2: Patch MAGMA for CUDA 13.1 compatibility
      # This fixes: 'struct cudaDeviceProp' has no member named 'clockRate'
      (final: prev: {
        magma = prev.magma.overrideAttrs (oldAttrs: {
          patches = (oldAttrs.patches or []) ++ [
            (final.fetchpatch {
              name = "cuda-13.0-clockrate-fix.patch";
              url = "https://github.com/icl-utk-edu/magma/commit/235aefb7b064954fce09d035c69907ba8a87cbcd.patch";
              hash = "sha256-i9InbxD5HtfonB/GyF9nQhFmok3jZ73RxGcIciGBGvU=";
            })
          ];
        });
      })

      # Overlay 3: Upgrade PyTorch to 2.10.0
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

  # GPU target: SM100 (Blackwell Datacenter - B100/B200)
  gpuArchSM = "10.0";

  # CPU optimization: AVX-512 BF16
  cpuFlags = [
    "-mavx512f"      # AVX-512 Foundation
    "-mavx512dq"     # Doubleword and Quadword instructions
    "-mavx512vl"     # Vector Length extensions
    "-mavx512bw"     # Byte and Word instructions
    "-mavx512bf16"   # BF16 (Brain Float 16) instructions
    "-mfma"          # Fused multiply-add
  ];

  # Custom PyTorch 2.10.0 with all CUDA 13.1 fixes
  customPytorch = (nixpkgs_pinned.python3Packages.torch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];
  }).overrideAttrs (oldAttrs: {
    pname = "pytorch210-for-torchvision-sm100-avx512bf16";

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
    pname = "torchvision-python313-cuda13_1-sm100-avx512bf16";

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
      echo "GPU Target: ${gpuArchSM} (Blackwell Datacenter: B100/B200)"
      echo "CPU Features: AVX-512 BF16"
      echo "CUDA: 13.1"
      echo "PyTorch: 2.10.0 (with CUDA 13.1 fixes)"
      echo "MAGMA: Enabled (with CUDA 13.1 patch)"
      echo "TorchVision: ${oldAttrs.version}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision for NVIDIA B100/B200 Datacenter (SM100, Blackwell) + AVX-512 BF16 with PyTorch 2.10.0";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA Blackwell Datacenter architecture (SM100) - B100/B200
        - CPU: x86-64 with AVX-512 BF16 instruction set
        - CUDA: 13.1 with compute capability 10.0
        - PyTorch: 2.10.0 (with all CUDA 13.1 compatibility fixes)
        - MAGMA: Enabled (patched for CUDA 13.1)
        - Python: 3.13

        Hardware requirements:
        - GPU: B100, B200, or other SM100 GPUs
        - CPU: Intel Cooper Lake+ (2020+), AMD Zen 4+ (2022+)
        - Driver: NVIDIA 550+ required
      '';
      platforms = [ "x86_64-linux" ];
    };
  })
