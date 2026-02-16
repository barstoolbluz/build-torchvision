# TorchVision with PyTorch 2.10.0 + MPS for Apple Silicon
# Package name: torchvision-python313-darwin-mps
#
# macOS build for Apple Silicon (M1/M2/M3/M4) with Metal GPU acceleration
# Requires: macOS 12.3+

{ pkgs ? import <nixpkgs> {} }:

let
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/6a030d535719c5190187c4cec156f335e95e3211.tar.gz";
  }) {
    config = { allowUnfree = true; allowBroken = true; };
    overlays = [
      (final: prev: {
        python3Packages = prev.python3Packages.override {
          overrides = pfinal: pprev: {
            torchvision = pprev.torchvision.overrideAttrs (oldAttrs: rec {
              version = "0.25.0";
              src = prev.fetchFromGitHub {
                owner = "pytorch";
                repo = "vision";
                rev = "v${version}";
                hash = "sha256-oktJHcT6T4f58pUO+HSBpbyS1ISH3zDlTsXQh6PcMy4=";
              };
            });
          };
        };
      })
    ];
  };

  # Import the exact pytorch derivation from build-pytorch
  # This ensures torchvision links against the same libtorch as the published pytorch package
  pytorch_src = builtins.fetchTarball {
    url = "https://github.com/barstoolbluz/build-pytorch/archive/6317465.tar.gz";
  };
  customPytorch = import "${pytorch_src}/.flox/pkgs/pytorch-python313-darwin-mps.nix" {};

in
  (nixpkgs_pinned.python3Packages.torchvision.override {
    torch = customPytorch;
  }).overrideAttrs (oldAttrs: {
    pname = "torchvision-python313-darwin-mps";

    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    buildInputs = nixpkgs_pinned.lib.filter (p: !(nixpkgs_pinned.lib.hasPrefix "cuda" (p.pname or "")))
      (oldAttrs.buildInputs or []);
    nativeBuildInputs = nixpkgs_pinned.lib.filter (p: p.pname or "" != "addDriverRunpath")
      (oldAttrs.nativeBuildInputs or []);

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export MAX_JOBS=32
      echo "MPS build | PyTorch: 2.10.0 | Platform: aarch64-darwin"
    '';

    postInstall = (oldAttrs.postInstall or "") + ''
      echo 1 > $out/.metadata-rev
    '';

    meta = oldAttrs.meta // {
      description = "TorchVision with MPS GPU acceleration for Apple Silicon";
      longDescription = ''
        Custom TorchVision build with targeted optimizations:
        - GPU: Metal Performance Shaders (MPS) for Apple Silicon
        - Platform: macOS 12.3+ on M1/M2/M3/M4
        - BLAS: vecLib (Apple Accelerate framework)
        - Python: 3.13
      '';
      platforms = [ "aarch64-darwin" ];
    };
  })
