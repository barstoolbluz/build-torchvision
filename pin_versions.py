#!/usr/bin/env python3
"""
Update all python313-cuda12_8 and python313-cpu variants to use
PyTorch 2.8.0 and TorchVision 0.23.0 for compatibility.
"""

import os
import re
import glob

def update_nix_file(filepath):
    """Update a single .nix file to use pinned versions."""

    # Skip our test files
    if any(x in filepath for x in ['-fixed.nix', '-compat.nix', '-v2.nix']):
        print(f"Skipping test file: {filepath}")
        return

    print(f"Updating: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already updated
    if 'pytorch_2_8_0' in content:
        print(f"  Already updated, skipping")
        return

    # Add fetchurl to imports if not present
    if 'fetchurl' not in content:
        content = content.replace(
            ', addDriverRunpath\n}:',
            ', addDriverRunpath\n, fetchurl\n}:'
        )
        # Also handle case without addDriverRunpath
        content = content.replace(
            ', openblas\n}:',
            ', openblas\n, fetchurl\n}:'
        )

    # Replace the customPytorch definition
    # Handle both GPU and CPU variants

    # Pattern for GPU variants
    gpu_pattern = r'(  # Custom PyTorch.*?\n.*?TODO.*?\n)?  customPytorch = \(python3Packages\.(?:pytorch|torch)\.override \{'
    gpu_replacement = '''  # Pin PyTorch to 2.8.0 for compatibility with TorchVision 0.23.0
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
  customPytorch = (pytorch_2_8_0.override {'''

    content = re.sub(gpu_pattern, gpu_replacement, content, flags=re.DOTALL)

    # Replace the torchvision definition
    torchvision_pattern = r'in\n  \(python3Packages\.torchvision\.override \{'
    torchvision_replacement = '''  # Pin TorchVision to 0.23.0
  torchvision_0_23_0 = python3Packages.torchvision.overrideAttrs (oldAttrs: rec {
    version = "0.23.0";
    pname = "torchvision";

    src = fetchurl {
      url = "https://github.com/pytorch/vision/archive/v${version}.tar.gz";
      hash = "sha256-1d09xwblldgzmzfdlrsyx6mgv939z4yi1hqanm9yx63cs2mr7w85";
    };
  });

in
  (torchvision_0_23_0.override {'''

    content = re.sub(torchvision_pattern, torchvision_replacement, content, flags=re.DOTALL)

    # Update echo statements to show pinned versions
    # Add version echo if not present
    if 'PyTorch Version:' not in content:
        content = re.sub(
            r'(echo "CUDA: .*?")',
            r'\1\n      echo "PyTorch Version: 2.8.0 (pinned)"\n      echo "TorchVision Version: 0.23.0 (pinned)"',
            content
        )

    # For CPU-only builds that don't have CUDA line
    if 'echo "CUDA: Disabled"' in content:
        content = content.replace(
            'echo "CUDA: Disabled"',
            'echo "CUDA: Disabled"\n      echo "PyTorch Version: 2.8.0 (pinned)"\n      echo "TorchVision Version: 0.23.0 (pinned)"'
        )

    # Update meta description to include version
    content = re.sub(
        r'description = "TorchVision for',
        'description = "TorchVision 0.23.0 for',
        content
    )
    content = re.sub(
        r'description = "TorchVision CPU-only',
        'description = "TorchVision 0.23.0 CPU-only',
        content
    )

    # Update longDescription to include versions
    if '- PyTorch: 2.8.0' not in content:
        content = re.sub(
            r'(- Python: 3.13)',
            r'\1\n        - PyTorch: 2.8.0 (pinned for compatibility)\n        - TorchVision: 0.23.0 (pinned)',
            content
        )

    # Write back the updated content
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✓ Updated successfully")

def main():
    # Update all cuda12_8 GPU variants
    gpu_files = glob.glob('.flox/pkgs/torchvision-python313-cuda12_8-*.nix')
    print(f"Found {len(gpu_files)} GPU variant files")

    for filepath in sorted(gpu_files):
        update_nix_file(filepath)

    # Update all CPU variants
    cpu_files = glob.glob('.flox/pkgs/torchvision-python313-cpu-*.nix')
    print(f"\nFound {len(cpu_files)} CPU variant files")

    for filepath in sorted(cpu_files):
        update_nix_file(filepath)

    print("\n✅ All files have been updated to use PyTorch 2.8.0 and TorchVision 0.23.0")

if __name__ == '__main__':
    main()