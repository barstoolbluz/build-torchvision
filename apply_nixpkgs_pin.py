#!/usr/bin/env python3
"""
Apply the nixpkgs pinning fix to all TorchVision nix expressions.
This ensures PyTorch 2.8.0 and TorchVision 0.23.0 compatibility.
"""

import os
import glob
import re

# The pinned nixpkgs configuration to insert
NIXPKGS_CONFIG = """  # Import nixpkgs at a specific revision where PyTorch 2.8.0 and TorchVision 0.23.0 are compatible
  # This commit has TorchVision 0.23.0 and PyTorch 2.8.0
  nixpkgs_pinned = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/fe5e41d7ffc0421f0913e8472ce6238ed0daf8e3.tar.gz";
    # You can add the sha256 here once known for reproducibility
  }) {
    config = {
      allowUnfree = true;  # Required for CUDA packages
      cudaSupport = true;
    };
  };"""

def should_skip_file(filepath):
    """Skip files that aren't TorchVision variants or are special cases."""
    basename = os.path.basename(filepath)
    skip_files = [
        'test-simple.nix',
        'torchvision-python313-cuda12_8-sm120-avx512-wrapped.nix',
        'torchvision-sm120-avx512-simple.nix',
        'torchvision-python313-cuda12_8-sm120-avx512.nix'  # Already updated
    ]
    return basename in skip_files

def update_nix_file(filepath):
    """Update a single nix file to use pinned nixpkgs."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already updated
    if 'nixpkgs_pinned' in content:
        print(f"✓ Already updated: {os.path.basename(filepath)}")
        return False

    # Replace pkgs.python3Packages with nixpkgs_pinned.python3Packages
    content = re.sub(r'\bpkgs\.python3Packages\b', 'nixpkgs_pinned.python3Packages', content)
    content = re.sub(r'\bpkgs\.python313Packages\b', 'nixpkgs_pinned.python313Packages', content)
    content = re.sub(r'\bpkgs\.python312Packages\b', 'nixpkgs_pinned.python312Packages', content)
    content = re.sub(r'\bpkgs\.python311Packages\b', 'nixpkgs_pinned.python311Packages', content)

    # Replace pkgs.lib with nixpkgs_pinned.lib
    content = re.sub(r'\bpkgs\.lib\b', 'nixpkgs_pinned.lib', content)

    # Find the let block and insert nixpkgs_pinned definition
    let_pattern = r'(\nlet\n)'
    if re.search(let_pattern, content):
        content = re.sub(let_pattern, r'\1' + NIXPKGS_CONFIG + '\n\n', content)
    else:
        print(f"⚠ Warning: Could not find 'let' block in {os.path.basename(filepath)}")
        return False

    # Write back the updated content
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✅ Updated: {os.path.basename(filepath)}")
    return True

def main():
    """Update all TorchVision nix expressions."""
    nix_files = glob.glob('/home/daedalus/dev/builds/build-torchvision/.flox/pkgs/*.nix')

    updated = 0
    skipped = 0
    already_updated = 0

    for filepath in sorted(nix_files):
        if should_skip_file(filepath):
            print(f"⊘ Skipping: {os.path.basename(filepath)}")
            skipped += 1
            continue

        result = update_nix_file(filepath)
        if result:
            updated += 1
        else:
            already_updated += 1

    print(f"\n📊 Summary:")
    print(f"  Updated: {updated}")
    print(f"  Already updated: {already_updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Total files: {len(nix_files)}")

if __name__ == "__main__":
    main()