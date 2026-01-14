# TorchVision optimized for NVIDIA DRIVE Thor (SM110) + AVX-512
# Package name: torchvision-python313-cuda12_8-sm110-avx512

{ python3Packages
, lib
, config
, cudaPackages
, addDriverRunpath
, fetchurl
}:

let
  # GPU target: SM110 (NVIDIA DRIVE Thor/Orin+ - Automotive/Edge)
  gpuArchNum = "110";        # For CMAKE_CUDA_ARCHITECTURES (just the integer)
  gpuArchSM = "sm_110";      # For TORCH_CUDA_ARCH_LIST (with sm_ prefix)

  # CPU optimization: AVX-512
  cpuFlags = [
    "-mavx512f"    # AVX-512 Foundation
    "-mavx512dq"   # Doubleword and Quadword instructions
    "-mavx512vl"   # Vector Length extensions
    "-mavx512bw"   # Byte and Word instructions
    "-mfma"        # Fused multiply-add
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
    pname = "torchvision-python313-cuda12_8-sm110-avx512";

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
      echo "GPU Target: SM110 (NVIDIA DRIVE Thor/Orin+)"
      echo "CPU Features: AVX-512"
      echo "CUDA: 12.8 (Compute Capability 11.0)"
      echo "PyTorch Version: 2.8.0 (pinned)"
      echo "TorchVision Version: 0.23.0 (pinned)"
      echo "CXXFLAGS: $CXXFLAGS"
      echo "Build parallelism: 32 cores max"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision 0.23.0 for NVIDIA DRIVE Thor (SM110) + AVX-512";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA DRIVE Thor/Orin+ (SM110) - Automotive/Edge AI
        - CPU: x86-64 with AVX-512 instruction set
        - CUDA: 12.8 with compute capability 11.0
        - Python: 3.13
        - PyTorch: 2.8.0 (pinned for compatibility)
        - TorchVision: 0.23.0 (pinned)
        - PyTorch: Custom build with matching GPU/CPU configuration

        Hardware requirements:
        - GPU: NVIDIA DRIVE Thor, Orin+ (Automotive/Embedded platforms)
        - CPU: Intel Skylake-X+ (2017+), AMD Zen 4+ (2022+)
        - Driver: NVIDIA 570+ required

        NOTE: This package depends on a matching PyTorch variant.
        Ensure pytorch-python313-cuda12_8-sm110-avx512 is installed.
      '';
      platforms = [ "x86_64-linux" ];
    };
  })
