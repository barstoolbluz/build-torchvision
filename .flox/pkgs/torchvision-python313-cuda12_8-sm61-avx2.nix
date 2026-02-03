# TorchVision optimized for NVIDIA Pascal GTX 1070/1080 Ti (SM61) + AVX2
# Package name: torchvision-python313-cuda12_8-sm61-avx2

{ pkgs ? import <nixpkgs> {} }:

let
  # Import nixpkgs at a specific revision where PyTorch 2.8.0 and TorchVision 0.23.0 are compatible
  # This commit has TorchVision 0.23.0 and PyTorch 2.8.0
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/fe5e41d7ffc0421f0913e8472ce6238ed0daf8e3.tar.gz";
  }) {
    config = {
      allowUnfree = true;  # Required for CUDA packages
      cudaSupport = true;
    };
  };

  # GPU target: SM61 (Pascal consumer - GTX 1070, 1080, 1080 Ti)
  gpuArchNum = "61";
  gpuArchSM = "6.1";  # Dot notation required for older archs in TORCH_CUDA_ARCH_LIST

  # CPU optimization: AVX2 (modern CPU with Pascal GPU)
  cpuFlags = [
    "-mavx2"
    "-mfma"
    "-mf16c"
  ];

  # Custom PyTorch with matching GPU/CPU configuration
  customPytorch = (nixpkgs_pinned.python3Packages.torch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];
  }).overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32

      # cuDNN 9.11+ dropped SM < 7.5 support — disable for SM61
      export USE_CUDNN=0
    '';
  });

in
  (nixpkgs_pinned.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda12_8-sm61-avx2";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${nixpkgs_pinned.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: ${gpuArchSM} (Pascal: GTX 1070, 1080, 1080 Ti)"
      echo "CPU Features: AVX2 (broad compatibility)"
      echo "CUDA: Enabled"
      echo "PyTorch: ${customPytorch.version}"
      echo "TorchVision: ${oldAttrs.version}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision for NVIDIA GTX 1070/1080 Ti (SM61, Pascal) with AVX2";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA Pascal consumer architecture (SM61) - GTX 1070, 1080, 1080 Ti
        - CPU: x86-64 with AVX2 instruction set (broad compatibility)
        - CUDA: 12.8 with compute capability 6.1
        - Python: 3.13
        - PyTorch: 2.8.0
        - TorchVision: 0.23.0

        Hardware requirements:
        - GPU: GTX 1070, 1080, 1080 Ti, or other SM61 GPUs
        - CPU: Intel Haswell+ (2013+), AMD Zen 1+ (2017+)
        - Driver: NVIDIA 390+ required

        Note: cuDNN is disabled because cuDNN 9.11+ dropped SM < 7.5 support.
        FBGEMM, MKLDNN, and NNPACK are enabled (AVX2+FMA3 available).
      '';
      platforms = [ "x86_64-linux" ];
    };
  })
