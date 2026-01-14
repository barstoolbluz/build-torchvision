#!/usr/bin/env bash
# Script to update all python313-cuda12_8 variants to use PyTorch 2.8.0 and TorchVision 0.23.0

set -e

# Create a function to update a single file
update_variant() {
    local file="$1"
    echo "Updating: $file"

    # Create backup
    cp "$file" "${file}.bak"

    # Replace the pytorch and torchvision definitions
    cat > "$file" << 'EOF'
# TorchVision 0.23.0 optimized build - pinned to PyTorch 2.8.0 for compatibility
# This file has been updated to use compatible versions

{ python3Packages
, lib
, config
, cudaPackages
, addDriverRunpath
, fetchFromGitHub
, fetchurl
}:

let
EOF

    # Extract the GPU and CPU configuration from the backup file
    # Get everything between "let" and "customPytorch"
    sed -n '/^let$/,/customPytorch = /p' "${file}.bak" | sed '1d;$d' | sed 's/^  //' >> "$file"

    # Add the new pytorch and customPytorch definitions
    cat >> "$file" << 'EOF'

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
EOF

    # Check if this is a CPU-only build or GPU build
    if grep -q "cudaSupport = false" "${file}.bak"; then
        echo "    cudaSupport = false;" >> "$file"
    else
        # Extract GPU targets line
        grep "gpuTargets = " "${file}.bak" | head -1 | sed 's/^  /    /' >> "$file"
        echo "    cudaSupport = true;" >> "$file"
    fi

    cat >> "$file" << 'EOF'
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
EOF

    # Get the rest of the file (pname onwards)
    sed -n '/^    pname = /,$p' "${file}.bak" | sed 's/^  //' >> "$file"

    # Update version mentions in the echo statements and meta
    sed -i 's/PyTorch Version: .*/PyTorch Version: 2.8.0 (pinned)"/g' "$file"
    sed -i 's/TorchVision Version: .*/TorchVision Version: 0.23.0 (pinned)"/g' "$file"
    sed -i '/Custom TorchVision build/a\        - PyTorch: 2.8.0 (pinned for compatibility)\n        - TorchVision: 0.23.0 (pinned)' "$file"

    echo "Updated: $file"
}

# Find all python313-cuda12_8 variant files
for file in .flox/pkgs/torchvision-python313-cuda12_8-*.nix; do
    if [[ -f "$file" && ! "$file" == *.bak ]]; then
        update_variant "$file"
    fi
done

# Also update CPU-only variants
for file in .flox/pkgs/torchvision-python313-cpu-*.nix; do
    if [[ -f "$file" && ! "$file" == *.bak ]]; then
        update_variant "$file"
    fi
done

echo "All variants have been updated to use PyTorch 2.8.0 and TorchVision 0.23.0"