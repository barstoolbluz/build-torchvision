# TorchVision optimized for NVIDIA DGX Spark (SM121) + ARMv9
# Package name: torchvision-python313-cuda12_8-sm121-armv9

{ python3Packages
, lib
, config
, cudaPackages
, addDriverRunpath
}:

let
  # GPU target: SM121 (DGX Spark - specialized datacenter)
  gpuArchNum = "121";        # For CMAKE_CUDA_ARCHITECTURES (just the integer)
  gpuArchSM = "sm_121";      # For TORCH_CUDA_ARCH_LIST (with sm_ prefix)

  # CPU optimization: ARMv9 with SVE2 and other advanced features
  cpuFlags = [
    "-march=armv9-a+sve2"  # ARMv9 with SVE2 (Scalable Vector Extension 2)
  ];

  # Custom PyTorch with matching GPU/CPU configuration
  # TODO: Reference the actual pytorch package from build-pytorch
  customPytorch = (python3Packages.pytorch.override {
    cudaSupport = true;
    gpuTargets = [ gpuArchSM ];
  }).overrideAttrs (oldAttrs: {
    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
    '';
  });

in
  (python3Packages.torchvision.override {
    pytorch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-cuda12_8-sm121-armv9";

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: SM121 (DGX Spark - Specialized Datacenter)"
      echo "CPU Features: ARMv9 + SVE2"
      echo "CUDA: 12.8 (Compute Capability 12.1)"
      echo "CXXFLAGS: $CXXFLAGS"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision for NVIDIA DGX Spark (SM121) + ARMv9";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: NVIDIA DGX Spark (SM121, Compute Capability 12.1)
        - CPU: ARMv9 with SVE2 (Scalable Vector Extension 2)
        - CUDA: 12.8
        - Python: 3.13

        Hardware requirements:
        - GPU: DGX Spark specialized datacenter GPUs
        - CPU: AWS Graviton3+, NVIDIA Grace, ARM Neoverse V1+
        - Driver: NVIDIA 570+ required

        Choose this if: You have DGX Spark in ARM-based datacenter with
        latest processors (Graviton3+, Grace Hopper superchip). ARMv9
        with SVE2 provides best performance for ARM platforms.
      '';
      platforms = [ "aarch64-linux" ];
    };
  })
