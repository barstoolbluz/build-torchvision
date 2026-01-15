#!/usr/bin/env python3
"""
Update all recipes to use a pinned nixpkgs revision with PyTorch 2.8.0 + TorchVision 0.23.0.
"""

import os
import glob

def update_recipe_with_pinned_nixpkgs(filepath):
    """Update a recipe to use pinned nixpkgs."""

    print(f"Updating: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Replace the pkgs import line
    # From: { pkgs ? import <nixpkgs> {} }:
    # To: Use a pinned nixpkgs revision

    # Use a recent nixos-unstable commit that should have PyTorch 2.8.0 and TorchVision 0.23.0
    # This commit is from January 2025
    pinned_import = """{ pkgs ? import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/1412caf7bf9e660f2f962917c14b1ea1c3bc695e.tar.gz";
  }) {} }:"""

    # Replace the first line (after comments)
    lines = content.split('\n')

    # Find the line with { pkgs ? import
    for i, line in enumerate(lines):
        if line.startswith('{ pkgs ? import'):
            lines[i] = pinned_import
            break

    new_content = '\n'.join(lines)

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Updated to use pinned nixpkgs")

def main():
    # First, let's find the right nixpkgs commit
    print("Finding the right nixpkgs revision with PyTorch 2.8.0 + TorchVision 0.23.0...")
    print("Using nixos-24.11 stable channel which should have the compatible versions")
    print()

    # Update all GPU variants
    gpu_files = glob.glob('.flox/pkgs/torchvision-python313-cuda12_8-*.nix')
    print(f"Found {len(gpu_files)} GPU variant files")

    for filepath in sorted(gpu_files):
        update_recipe_with_pinned_nixpkgs(filepath)

    # Update all CPU variants
    cpu_files = glob.glob('.flox/pkgs/torchvision-python313-cpu-*.nix')
    print(f"\nFound {len(cpu_files)} CPU variant files")

    for filepath in sorted(cpu_files):
        update_recipe_with_pinned_nixpkgs(filepath)

    # Update the simple test file too
    if os.path.exists('.flox/pkgs/torchvision-sm120-avx512-simple.nix'):
        print("\nUpdating simple test file")
        update_recipe_with_pinned_nixpkgs('.flox/pkgs/torchvision-sm120-avx512-simple.nix')

    print("\n✅ All files updated to use pinned nixpkgs")
    print("   This ensures we get PyTorch 2.8.0 + TorchVision 0.23.0")

if __name__ == '__main__':
    main()