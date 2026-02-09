# TorchVision with MPS (Metal Performance Shaders) for Apple Silicon
# Package name: torchvision-python313-darwin-mps
#
# macOS build for Apple Silicon (M1/M2/M3/M4) with Metal GPU acceleration
# Hardware: Apple M1, M2, M3, M4 and variants (Pro, Max, Ultra)
# Requires: macOS 12.3+
#
# Note: Frameworks (Metal, Accelerate, etc.) are provided automatically by
# apple-sdk_13 via $SDKROOT — no individual framework packages needed.

{ pkgs ? import <nixpkgs> {} }:

let
  # Import nixpkgs at a specific revision where PyTorch 2.8.0 and TorchVision 0.23.0 are compatible
  # This commit has TorchVision 0.23.0 and PyTorch 2.8.0
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/fe5e41d7ffc0421f0913e8472ce6238ed0daf8e3.tar.gz";
    # You can add the sha256 here once known for reproducibility
  }) {
    config = {
      allowUnfree = true;
      cudaSupport = false;
    };
  };

  # Custom PyTorch with MPS configuration for Apple Silicon
  customPytorch = (nixpkgs_pinned.python3Packages.torch.override {
    cudaSupport = false;
  }).overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    # Filter out CUDA deps (base pytorch may include them)
    buildInputs = nixpkgs_pinned.lib.filter (p: !(nixpkgs_pinned.lib.hasPrefix "cuda" (p.pname or "")))
      (oldAttrs.buildInputs or []);
    nativeBuildInputs = nixpkgs_pinned.lib.filter (p: p.pname or "" != "addDriverRunpath")
      (oldAttrs.nativeBuildInputs or []);

    preConfigure = (oldAttrs.preConfigure or "") + ''
      # Disable CUDA
      export USE_CUDA=0
      export USE_CUDNN=0
      export USE_CUBLAS=0

      # Enable MPS (Metal Performance Shaders)
      export USE_MPS=1
      export USE_METAL=1

      # Use vecLib (Apple Accelerate) for BLAS
      export BLAS=vecLib
      export MAX_JOBS=32
    '';
  });

in
  (nixpkgs_pinned.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-darwin-mps";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    # Filter out any CUDA deps from torchvision
    buildInputs = nixpkgs_pinned.lib.filter (p: !(nixpkgs_pinned.lib.hasPrefix "cuda" (p.pname or "")))
      (oldAttrs.buildInputs or []);
    nativeBuildInputs = nixpkgs_pinned.lib.filter (p: p.pname or "" != "addDriverRunpath")
      (oldAttrs.nativeBuildInputs or []);

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: MPS (Metal Performance Shaders)"
      echo "Platform: Apple Silicon (aarch64-darwin)"
      echo "BLAS Backend: vecLib (Apple Accelerate)"
      echo "PyTorch: ${customPytorch.version}"
      echo "TorchVision: ${oldAttrs.version}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision with MPS GPU acceleration for Apple Silicon";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: Metal Performance Shaders (MPS) for Apple Silicon
        - Platform: macOS 12.3+ on M1/M2/M3/M4
        - BLAS: vecLib (Apple Accelerate framework)
        - PyTorch: 2.8.0 with MPS support
        - Python: 3.13
      '';
      platforms = [ "aarch64-darwin" ];
    };
  })
