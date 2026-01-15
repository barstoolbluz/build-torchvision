#!/usr/bin/env python3
"""
Update all recipes to use local nixpkgs (which has PyTorch 2.8.0 + TorchVision 0.23.0).
"""

import os
import glob

def update_recipe_to_local_nixpkgs(filepath):
    """Update a recipe to use local nixpkgs."""

    print(f"Updating: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Replace the pinned nixpkgs import with local nixpkgs
    # From: { pkgs ? import (fetchTarball { url = "..."; }) {} }:
    # To: { pkgs ? import <nixpkgs> {} }:

    lines = content.split('\n')

    # Find and replace the import line
    for i, line in enumerate(lines):
        if '{ pkgs ? import (fetchTarball' in line:
            # Skip to the closing of the import statement
            j = i
            while j < len(lines) and '}) {} }:' not in lines[j]:
                j += 1
            # Replace with simple import
            lines[i:j+1] = ['{ pkgs ? import <nixpkgs> {} }:']
            break
        elif '{ pkgs ? import <nixpkgs>' in line:
            # Already using local nixpkgs
            print(f"  ✓ Already using local nixpkgs")
            return

    new_content = '\n'.join(lines)

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Updated to use local nixpkgs")

def main():
    print("Updating all recipes to use local nixpkgs...")
    print("Your local nixpkgs has PyTorch 2.8.0 + TorchVision 0.23.0")
    print()

    # Update all variants
    all_files = glob.glob('.flox/pkgs/torchvision-python313-*.nix')
    all_files.extend(glob.glob('.flox/pkgs/torchvision-sm120-*.nix'))

    print(f"Found {len(all_files)} recipe files")

    for filepath in sorted(set(all_files)):
        update_recipe_to_local_nixpkgs(filepath)

    print("\n✅ All files updated to use local nixpkgs")
    print("   This will use your system's PyTorch 2.8.0 + TorchVision 0.23.0")

if __name__ == '__main__':
    main()