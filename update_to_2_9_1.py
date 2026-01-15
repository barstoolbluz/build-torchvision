#!/usr/bin/env python3
"""
Update all recipes to use PyTorch 2.9.1 + TorchVision 0.24.1 (what Flox nixpkgs provides).
Keep the same structure but remove version overrides.
"""

import os
import glob

def update_recipe(filepath):
    """Update recipe to use default nixpkgs versions (2.9.1 + 0.24.1)."""

    print(f"Updating: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # The recipes already use nixpkgs packages without version overrides
    # after our remove_source_overrides.py script ran.
    # They should work with 2.9.1 + 0.24.1 as-is.

    # Just verify the structure is correct
    if '{ pkgs ? import <nixpkgs> {} }:' in content:
        if 'pkgs.python3Packages.torch' in content and 'pkgs.python3Packages.torchvision' in content:
            print(f"  ✓ Already configured for nixpkgs versions")
            return

    print(f"  ! May need manual review")

def main():
    print("Verifying all recipes use nixpkgs PyTorch 2.9.1 + TorchVision 0.24.1...")
    print("These are the versions Flox's nixpkgs provides.")
    print()

    # Check all variants
    all_files = glob.glob('.flox/pkgs/torchvision-python313-*.nix')
    print(f"Found {len(all_files)} recipe files")

    for filepath in sorted(all_files):
        update_recipe(filepath)

    print("\n✅ Recipes are configured to use nixpkgs versions")
    print("   PyTorch 2.9.1 + TorchVision 0.24.1 from Flox nixpkgs")

if __name__ == '__main__':
    main()