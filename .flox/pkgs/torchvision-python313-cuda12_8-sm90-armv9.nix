# TorchVision optimized for NVIDIA Hopper H100/L40S (SM90) + ARMv9
# Package name: torchvision-python313-cuda12_8-sm90-armv9

{ pkgs ? import <nixpkgs> {} }:

let
  # GPU target
  gpuArchNum = "90";
  gpuArchSM = "sm_90";

  # CPU optimization
  cpuFlags = [
    "-march=armv9-a+sve+sve2"
  ];

  # Custom PyTorch with matching GPU/CPU configuration
  customPytorch = (pkgs.python3Packages.torch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];
  }).overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${pkgs.lib.concatStringsSep " " cpuFlags} $CXXFLAGS"
      export CFLAGS="${pkgs.lib.concatStringsSep " " cpuFlags} $CFLAGS"
      export MAX_JOBS=32
    '';
  });

in
  (pkgs.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda12_8-sm90-armv9";

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
      echo "GPU Target: sm_90"
      echo "CPU Features: Optimized"
      echo "CUDA: Enabled"
      echo "PyTorch: ${customPytorch.version}"
      echo "TorchVision: ${oldAttrs.version}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision optimized for NVIDIA Hopper H100/L40S (SM90) + ARMv9";
      platforms = oldAttrs.meta.platforms or [ "x86_64-linux" "aarch64-linux" ];
    };
  })
