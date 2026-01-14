# Simplified TorchVision for NVIDIA Blackwell (SM120: RTX 5090) + AVX-512
{ pkgs ? import <nixpkgs> {} }:

let
  # GPU target configuration
  gpuArchNum = "12.0";

  # CPU optimization flags
  cpuFlags = [
    "-mavx512f"
    "-mavx512dq"
    "-mavx512vl"
    "-mavx512bw"
    "-mfma"
  ];

  # Custom PyTorch with GPU/CPU configuration
  customPytorch = (pkgs.python3Packages.torch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchNum ];
  }).overrideAttrs (oldAttrs: {
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${pkgs.lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${pkgs.lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32
    '';
  });

in
  (pkgs.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-sm120-avx512-simple";

    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${pkgs.lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${pkgs.lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: SM120 (Blackwell: RTX 5090)"
      echo "CPU Features: AVX-512"
      echo "CUDA: 12.8 (Compute Capability 12.0)"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision for RTX 5090 (SM120) + AVX-512";
    };
  })