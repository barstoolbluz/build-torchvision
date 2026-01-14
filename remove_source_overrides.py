#!/usr/bin/env python3
"""
Remove source overrides from all TorchVision recipes.
Keep the custom build flags but use nixpkgs versions (PyTorch 2.8.0 + TorchVision 0.23.0).
"""

import os
import re
import glob

def update_recipe(filepath):
    """Update a single recipe to remove source overrides."""

    print(f"Updating: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Backup original
    with open(filepath + '.backup', 'w') as f:
        f.write(content)

    # Check if it's a GPU or CPU variant
    is_cpu = 'cpu' in os.path.basename(filepath)

    if is_cpu:
        # For CPU variants, create a simpler structure
        new_content = create_cpu_recipe(filepath, content)
    else:
        # For GPU variants
        new_content = create_gpu_recipe(filepath, content)

    # Write updated content
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Updated successfully")

def extract_metadata(content, filepath):
    """Extract key metadata from the original file."""

    # Extract GPU arch info (if present)
    gpu_arch_num_match = re.search(r'gpuArchNum = "([^"]+)"', content)
    gpu_arch_sm_match = re.search(r'gpuArchSM = "([^"]+)"', content)

    # Extract CPU flags
    cpu_flags = []
    flags_match = re.search(r'cpuFlags = \[(.*?)\];', content, re.DOTALL)
    if flags_match:
        flags_str = flags_match.group(1)
        cpu_flags = re.findall(r'"([^"]+)"', flags_str)

    # Extract package name from filename
    basename = os.path.basename(filepath)
    package_name = basename.replace('.nix', '')

    # Extract descriptions from content
    desc_match = re.search(r'# (.+?)\n# Package name:', content)
    description = desc_match.group(1) if desc_match else "TorchVision build"

    return {
        'package_name': package_name,
        'description': description,
        'gpu_arch_num': gpu_arch_num_match.group(1) if gpu_arch_num_match else None,
        'gpu_arch_sm': gpu_arch_sm_match.group(1) if gpu_arch_sm_match else None,
        'cpu_flags': cpu_flags
    }

def create_gpu_recipe(filepath, original_content):
    """Create a GPU recipe without source overrides."""

    meta = extract_metadata(original_content, filepath)

    # Determine which GPU pattern to use
    use_arch_sm = meta['gpu_arch_sm'] is not None

    gpu_target_setup = ""
    if use_arch_sm:
        gpu_target_setup = f'''
  # GPU target
  gpuArchNum = "{meta['gpu_arch_num']}";
  gpuArchSM = "{meta['gpu_arch_sm']}";'''
        gpu_targets = "[ gpuArchSM ]"
    else:
        gpu_target_setup = f'''
  # GPU target
  gpuArchNum = "{meta['gpu_arch_num']}";'''
        gpu_targets = "[ gpuArchNum ]"

    cpu_flags_str = '\n    '.join([f'"{flag}"' for flag in meta['cpu_flags']])

    return f'''# {meta['description']}
# Package name: {meta['package_name']}

{{ pkgs ? import <nixpkgs> {{}} }}:

let{gpu_target_setup}

  # CPU optimization
  cpuFlags = [
    {cpu_flags_str}
  ];

  # Custom PyTorch with matching GPU/CPU configuration
  customPytorch = (pkgs.python3Packages.torch.override {{
    cudaSupport = true;
    gpuTargets = {gpu_targets};
  }}).overrideAttrs (oldAttrs: {{
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CXXFLAGS"
      export CFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CFLAGS"
      export MAX_JOBS=32
    '';
  }});

in
  (pkgs.python3Packages.torchvision.override {{
    torch = customPytorch;
  }}).overrideAttrs (oldAttrs: {{
    pname = "{meta['package_name']}";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CXXFLAGS"
      export CFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CFLAGS"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: {meta['gpu_arch_sm'] or meta['gpu_arch_num']}"
      echo "CPU Features: Optimized"
      echo "CUDA: Enabled"
      echo "PyTorch: ${{customPytorch.version}}"
      echo "TorchVision: ${{oldAttrs.version}}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {{
      description = "{meta['description']}";
      platforms = oldAttrs.meta.platforms or [ "x86_64-linux" "aarch64-linux" ];
    }};
  }})
'''

def create_cpu_recipe(filepath, original_content):
    """Create a CPU recipe without source overrides."""

    meta = extract_metadata(original_content, filepath)

    cpu_flags_str = '\n    '.join([f'"{flag}"' for flag in meta['cpu_flags']])

    # Determine platform based on CPU ISA
    platform = '"aarch64-linux"' if 'arm' in meta['package_name'] else '"x86_64-linux"'

    return f'''# {meta['description']}
# Package name: {meta['package_name']}

{{ pkgs ? import <nixpkgs> {{}} }}:

let
  # CPU optimization
  cpuFlags = [
    {cpu_flags_str}
  ];

  # Custom PyTorch with CPU-only configuration
  customPytorch = (pkgs.python3Packages.torch.override {{
    cudaSupport = false;
  }}).overrideAttrs (oldAttrs: {{
    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    buildInputs = pkgs.lib.filter (p: !(pkgs.lib.hasPrefix "cuda" (p.pname or ""))) oldAttrs.buildInputs
                  ++ [pkgs.openblas];
    nativeBuildInputs = pkgs.lib.filter (p: p.pname or "" != "addDriverRunpath") oldAttrs.nativeBuildInputs;

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export USE_CUDA=0
      export BLAS=OpenBLAS
      export CXXFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CXXFLAGS"
      export CFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CFLAGS"
      export MAX_JOBS=32
    '';
  }});

in
  (pkgs.python3Packages.torchvision.override {{
    torch = customPytorch;
  }}).overrideAttrs (oldAttrs: {{
    pname = "{meta['package_name']}";

    # Limit build parallelism to prevent memory saturation
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CXXFLAGS"
      export CFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CFLAGS"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: None (CPU-only build)"
      echo "CPU Features: Optimized"
      echo "BLAS Backend: OpenBLAS"
      echo "PyTorch: ${{customPytorch.version}}"
      echo "TorchVision: ${{oldAttrs.version}}"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {{
      description = "{meta['description']}";
      platforms = [ {platform} ];
    }};
  }})
'''

def main():
    # Update all GPU variants
    gpu_files = glob.glob('.flox/pkgs/torchvision-python313-cuda12_8-*.nix')
    print(f"Found {len(gpu_files)} GPU variant files")

    for filepath in sorted(gpu_files):
        update_recipe(filepath)

    # Update all CPU variants
    cpu_files = glob.glob('.flox/pkgs/torchvision-python313-cpu-*.nix')
    print(f"\nFound {len(cpu_files)} CPU variant files")

    for filepath in sorted(cpu_files):
        update_recipe(filepath)

    print("\n✅ All files updated to use nixpkgs versions without source overrides")
    print("   PyTorch 2.8.0 and TorchVision 0.23.0 from nixpkgs will be used")
    print("   Custom build flags are preserved")

if __name__ == '__main__':
    main()