# TorchVision optimized for NVIDIA Blackwell (SM120: RTX 5090) + AVX-512
# Package name: torchvision-python313-cuda12_8-sm120-avx512

{ python3Packages
, lib
, config
, cudaPackages
, addDriverRunpath
}:

let
  # GPU target: SM120 (Blackwell architecture - RTX 5090)
  # PyTorch's CMake accepts numeric format (12.0) not sm_120
  gpuArchNum = "12.0";

  # CPU optimization: AVX-512
  cpuFlags = [
    "-mavx512f"    # AVX-512 Foundation
    "-mavx512dq"   # Doubleword and Quadword instructions
    "-mavx512vl"   # Vector Length extensions
    "-mavx512bw"   # Byte and Word instructions
    "-mfma"        # Fused multiply-add
  ];

  # Custom PyTorch with matching GPU/CPU configuration
  # TODO: Reference the actual pytorch package from build-pytorch
  customPytorch = (python3Packages.pytorch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchNum ];
  }).overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32
    '';
  });

in
  (python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda12_8-sm120-avx512";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: SM120 (Blackwell: RTX 5090)"
      echo "CPU Features: AVX-512"
      echo "CUDA: 12.8 (Compute Capability 12.0)"
      echo "CXXFLAGS: $CXXFLAGS"
      echo "Build parallelism: 32 cores max"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision for NVIDIA RTX 5090 (SM120, Blackwell) + AVX-512";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA Blackwell architecture (SM120) - RTX 5090
        - CPU: x86-64 with AVX-512 instruction set
        - CUDA: 12.8
        - Python: 3.13

        Hardware requirements:
        - GPU: RTX 5090, Blackwell architecture GPUs
        - CPU: Intel Skylake-X+ (2017+), AMD Zen 4+ (2022+)
        - Driver: NVIDIA 570+ required

        Choose this if: You have RTX 5090 with modern datacenter CPUs.
        For specialized workloads, consider avx512bf16 (BF16 training) or
        avx512vnni (INT8 inference) variants.
      '';
      platforms = [ "x86_64-linux" ];
    };
  })
