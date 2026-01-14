# TorchVision CPU-only optimized for AVX-512
# Package name: torchvision-python313-cpu-avx512

{ pkgs ? import <nixpkgs> {} }:

let
  # CPU optimization
  cpuFlags = [
    "-mavx512f"
    "-mavx512dq"
    "-mavx512vl"
    "-mavx512bw"
    "-mfma"
  ];

  # Custom PyTorch with CPU-only configuration
  customPytorch = (pkgs.python3Packages.torch.override {
    cudaSupport = false;
  }).overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    buildInputs = pkgs.lib.filter (p: !(pkgs.lib.hasPrefix "cuda" (p.pname or ""))) oldAttrs.buildInputs
                  ++ [pkgs.openblas];
    nativeBuildInputs = pkgs.lib.filter (p: p.pname or "" != "addDriverRunpath") oldAttrs.nativeBuildInputs;

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export USE_CUDA=0
      export BLAS=OpenBLAS
      export CXXFLAGS="${pkgs.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${pkgs.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32
    '';
  });

in
  (pkgs.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cpu-avx512";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${pkgs.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${pkgs.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: None (CPU-only build)"
      echo "CPU Features: Optimized"
      echo "BLAS Backend: OpenBLAS"
      echo "PyTorch: ${customPytorch.version}"
      echo "TorchVision: ${oldAttrs.version}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision CPU-only optimized for AVX-512";
      platforms = [ "x86_64-linux" ];
    };
  })
