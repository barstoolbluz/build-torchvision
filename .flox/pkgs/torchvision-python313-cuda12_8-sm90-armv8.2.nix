# TorchVision optimized for NVIDIA Hopper H100/L40S (SM90) + ARMv8.2
# Package name: torchvision-python313-cuda12_8-sm90-armv8.2

{ python3Packages
, lib
, config
, cudaPackages
, addDriverRunpath
, fetchurl
}:

let
  # GPU target: SM90 (Hopper H100/L40S)
  gpuArchNum = "90";        # For CMAKE_CUDA_ARCHITECTURES (just the integer)
  gpuArchSM = "sm_90";      # For TORCH_CUDA_ARCH_LIST (with sm_ prefix)

  # CPU optimization: ARMv8.2 with FP16 and dot product
  cpuFlags = [
    "-march=armv8.2-a+fp16+dotprod"  # ARMv8.2 with half-precision and dot product
  ];

  # Pin PyTorch to 2.8.0 for compatibility with TorchVision 0.23.0
  pytorch_2_8_0 = python3Packages.torch.overrideAttrs (oldAttrs: rec {
    version = "2.8.0";
    pname = "torch";

    # Override the source to use PyTorch 2.8.0
    src = fetchurl {
      url = "https://github.com/pytorch/pytorch/archive/v${version}.tar.gz";
      hash = "sha256-0am8mx0mq3hqsk1g99a04a4fdf865g93568qr1f247pl11r2jldl";
    };
  });

  # Custom PyTorch with matching GPU/CPU configuration
  customPytorch = (pytorch_2_8_0.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];
  }).overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32
    '';
  });

  # Pin TorchVision to 0.23.0
  torchvision_0_23_0 = python3Packages.torchvision.overrideAttrs (oldAttrs: rec {
    version = "0.23.0";
    pname = "torchvision";

    src = fetchurl {
      url = "https://github.com/pytorch/vision/archive/v${version}.tar.gz";
      hash = "sha256-1d09xwblldgzmzfdlrsyx6mgv939z4yi1hqanm9yx63cs2mr7w85";
    };
  });

in
  (torchvision_0_23_0.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda12_8-sm90-armv8.2";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: SM90 (Hopper H100/L40S)"
      echo "CPU Features: ARMv8.2 + FP16 + DotProd"
      echo "CUDA: 12.8 (Compute Capability 9.0)"
      echo "PyTorch Version: 2.8.0 (pinned)"
      echo "TorchVision Version: 0.23.0 (pinned)"
      echo "CXXFLAGS: $CXXFLAGS"
      echo "Build parallelism: 32 cores max"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision 0.23.0 for NVIDIA Hopper H100/L40S (SM90) + ARMv8.2";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA Hopper H100/L40S (SM90)
        - CPU: ARMv8.2 with FP16 and dot product instructions
        - CUDA: 12.8 with compute capability 9.0
        - Python: 3.13
        - PyTorch: 2.8.0 (pinned for compatibility)
        - TorchVision: 0.23.0 (pinned)
        - PyTorch: Custom build with matching GPU/CPU configuration

        Hardware requirements:
        - GPU: NVIDIA H100, H800, L40S
        - CPU: AWS Graviton2, NVIDIA Grace platforms
        - Driver: NVIDIA 525+ required

        Optimized for ARM-based platforms with Hopper GPUs.
      '';
      platforms = [ "aarch64-linux" ];
    };
  })
