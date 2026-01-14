# TorchVision CPU-only optimized for AVX2
# Package name: torchvision-python313-cpu-avx2

{ python3Packages
, lib
, openblas
, fetchurl
}:

let
  # CPU optimization: AVX2 (no GPU)
  cpuFlags = [
    "-mavx2"       # AVX2 instructions
    "-mfma"        # Fused multiply-add
    "-mf16c"       # Half-precision conversions
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

  # Custom PyTorch with matching CPU configuration
  customPytorch = (pytorch_2_8_0.overrideAttrs (oldAttrs: {
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    buildInputs = lib.filter (p: !(lib.hasPrefix "cuda" (p.pname or ""))) oldAttrs.buildInputs ++ [openblas];
    nativeBuildInputs = lib.filter (p: p.pname or "" != "addDriverRunpath") oldAttrs.nativeBuildInputs;

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export USE_CUDA=0
      export BLAS=OpenBLAS
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32
    '';
  }));

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
    pname = "torchvision-python313-cpu-avx2";
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: None (CPU-only build)"
      echo "CPU Features: AVX2"
      echo "BLAS Backend: OpenBLAS"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision 0.23.0 CPU-only optimized for AVX2";
      platforms = [ "x86_64-linux" ];
    };
  })
