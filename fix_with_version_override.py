#!/usr/bin/env python3
"""
Fix recipes by overriding PyTorch and TorchVision versions explicitly.
Since we can't find the exact nixpkgs commit with 2.8.0 + 0.23.0,
we'll override the versions explicitly.
"""

import os
import glob

def create_gpu_recipe_with_override(filepath, original_content):
    """Create a GPU recipe with explicit version overrides."""

    # Extract metadata
    basename = os.path.basename(filepath)
    package_name = basename.replace('.nix', '')

    # Determine GPU architecture pattern
    is_type_a = any(arch in basename for arch in ['sm121', 'sm110', 'sm103', 'sm100', 'sm90', 'sm89', 'sm80'])

    # Extract GPU arch info
    import re
    gpu_num = None
    gpu_sm = None

    if is_type_a:
        # Extract both values for Type A
        for arch in ['121', '110', '103', '100', '90', '89', '80']:
            if f'sm{arch}' in basename:
                gpu_num = {
                    '121': '12.1', '110': '11.0', '103': '10.3',
                    '100': '10.0', '90': '9.0', '89': '8.9', '80': '8.0'
                }.get(arch)
                gpu_sm = f'sm_{arch}'
                break
    else:
        # Type B patterns
        for arch in ['120', '86']:
            if f'sm{arch}' in basename:
                gpu_num = '12.0' if arch == '120' else '8.6'
                break

    # Extract CPU flags from filename
    cpu_flags = []
    if 'avx512vnni' in basename:
        cpu_flags = ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mavx512vnni", "-mfma"]
    elif 'avx512bf16' in basename:
        cpu_flags = ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mavx512bf16", "-mfma"]
    elif 'avx512' in basename:
        cpu_flags = ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mfma"]
    elif 'avx2' in basename:
        cpu_flags = ["-mavx2", "-mfma", "-mf16c"]
    elif 'armv9' in basename:
        cpu_flags = ["-march=armv9-a", "-mtune=native"]
    elif 'armv8' in basename:
        cpu_flags = ["-march=armv8.2-a", "-mtune=native"]

    # Build GPU targets section
    if gpu_sm:
        gpu_target_setup = f'''
  # GPU target
  gpuArchNum = "{gpu_num}";
  gpuArchSM = "{gpu_sm}";'''
        gpu_targets = '[ gpuArchSM ]'
    else:
        gpu_target_setup = f'''
  # GPU target
  gpuArchNum = "{gpu_num}";'''
        gpu_targets = '[ gpuArchNum ]'

    cpu_flags_str = '\n    '.join([f'"{flag}"' for flag in cpu_flags])

    # Determine description
    gpu_desc = gpu_sm.replace('_', '') if gpu_sm else f"sm{gpu_num.replace('.', '')}"
    cpu_desc = basename.split('-')[-1].replace('.nix', '').upper()

    return f'''# TorchVision optimized build
# Package name: {package_name}

{{ pkgs ? import <nixpkgs> {{}} }}:

let{gpu_target_setup}

  # CPU optimization
  cpuFlags = [
    {cpu_flags_str}
  ];

  # Override PyTorch to version 2.8.0 with custom configuration
  customPytorch = (pkgs.python3Packages.torch.override {{
    cudaSupport = true;
    gpuTargets = {gpu_targets};
  }}).overrideAttrs (oldAttrs: {{
    version = "2.8.0";
    src = pkgs.fetchFromGitHub {{
      owner = "pytorch";
      repo = "pytorch";
      rev = "v2.8.0";
      hash = "sha256-0am8mx0mq3hqsk1g99a04a4fdf865g93568qr1f247pl11r2jldl";
      fetchSubmodules = true;
    }};

    # Limit build parallelism
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
    pname = "{package_name}";
    version = "0.23.0";

    src = pkgs.fetchFromGitHub {{
      owner = "pytorch";
      repo = "vision";
      rev = "v0.23.0";
      hash = "sha256-1d09xwblldgzmzfdlrsyx6mgv939z4yi1hqanm9yx63cs2mr7w85";
      fetchSubmodules = true;
    }};

    # Limit build parallelism
    ninjaFlags = [ "-j32" ];
    requiredSystemFeatures = [ "big-parallel" ];

    preConfigure = (oldAttrs.preConfigure or "") + ''
      export CXXFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CXXFLAGS"
      export CFLAGS="${{pkgs.lib.concatStringsSep " " cpuFlags}} $CFLAGS"
      export MAX_JOBS=32

      echo "========================================="
      echo "TorchVision Build Configuration"
      echo "========================================="
      echo "GPU Target: {gpu_desc}"
      echo "CPU Features: {cpu_desc}"
      echo "CUDA: Enabled"
      echo "PyTorch: 2.8.0"
      echo "TorchVision: 0.23.0"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {{
      description = "TorchVision for {gpu_desc} + {cpu_desc}";
      platforms = oldAttrs.meta.platforms or [ "x86_64-linux" "aarch64-linux" ];
    }};
  }})
'''

def create_cpu_recipe_with_override(filepath, original_content):
    """Create a CPU recipe with explicit version overrides."""

    basename = os.path.basename(filepath)
    package_name = basename.replace('.nix', '')

    # Extract CPU flags
    cpu_flags = []
    if 'avx512vnni' in basename:
        cpu_flags = ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mavx512vnni", "-mfma"]
    elif 'avx512bf16' in basename:
        cpu_flags = ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mavx512bf16", "-mfma"]
    elif 'avx512' in basename:
        cpu_flags = ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mavx512bw", "-mfma"]
    elif 'avx2' in basename:
        cpu_flags = ["-mavx2", "-mfma", "-mf16c"]
    elif 'armv9' in basename:
        cpu_flags = ["-march=armv9-a", "-mtune=native"]
    elif 'armv8' in basename:
        cpu_flags = ["-march=armv8.2-a", "-mtune=native"]

    cpu_flags_str = '\n    '.join([f'"{flag}"' for flag in cpu_flags])
    platform = '"aarch64-linux"' if 'arm' in basename else '"x86_64-linux"'
    cpu_desc = basename.split('-')[-1].replace('.nix', '').upper()

    return f'''# TorchVision CPU-only optimized build
# Package name: {package_name}

{{ pkgs ? import <nixpkgs> {{}} }}:

let
  # CPU optimization
  cpuFlags = [
    {cpu_flags_str}
  ];

  # Override PyTorch to version 2.8.0 with CPU-only configuration
  customPytorch = (pkgs.python3Packages.torch.override {{
    cudaSupport = false;
  }}).overrideAttrs (oldAttrs: {{
    version = "2.8.0";
    src = pkgs.fetchFromGitHub {{
      owner = "pytorch";
      repo = "pytorch";
      rev = "v2.8.0";
      hash = "sha256-0am8mx0mq3hqsk1g99a04a4fdf865g93568qr1f247pl11r2jldl";
      fetchSubmodules = true;
    }};

    # Limit build parallelism
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
    pname = "{package_name}";
    version = "0.23.0";

    src = pkgs.fetchFromGitHub {{
      owner = "pytorch";
      repo = "vision";
      rev = "v0.23.0";
      hash = "sha256-1d09xwblldgzmzfdlrsyx6mgv939z4yi1hqanm9yx63cs2mr7w85";
      fetchSubmodules = true;
    }};

    # Limit build parallelism
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
      echo "CPU Features: {cpu_desc}"
      echo "BLAS Backend: OpenBLAS"
      echo "PyTorch: 2.8.0"
      echo "TorchVision: 0.23.0"
      echo "========================================="
    '';

    meta = oldAttrs.meta // {{
      description = "TorchVision CPU-only for {cpu_desc}";
      platforms = [ {platform} ];
    }};
  }})
'''

def update_recipe(filepath):
    """Update a single recipe."""

    print(f"Updating: {filepath}")

    with open(filepath, 'r') as f:
        original_content = f.read()

    # Backup
    with open(filepath + '.backup2', 'w') as f:
        f.write(original_content)

    # Check if CPU or GPU variant
    is_cpu = 'cpu' in os.path.basename(filepath)

    if is_cpu:
        new_content = create_cpu_recipe_with_override(filepath, original_content)
    else:
        new_content = create_gpu_recipe_with_override(filepath, original_content)

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  ✓ Updated with explicit version overrides")

def main():
    print("Updating all recipes with explicit PyTorch 2.8.0 + TorchVision 0.23.0 overrides...")
    print()

    # Update GPU variants
    gpu_files = glob.glob('.flox/pkgs/torchvision-python313-cuda12_8-*.nix')
    print(f"Found {len(gpu_files)} GPU variant files")

    for filepath in sorted(gpu_files):
        # Skip wrapped file
        if 'wrapped' in filepath:
            print(f"Skipping: {filepath}")
            continue
        update_recipe(filepath)

    # Update CPU variants
    cpu_files = glob.glob('.flox/pkgs/torchvision-python313-cpu-*.nix')
    print(f"\nFound {len(cpu_files)} CPU variant files")

    for filepath in sorted(cpu_files):
        update_recipe(filepath)

    print("\n✅ All files updated with explicit version overrides")
    print("   PyTorch 2.8.0 and TorchVision 0.23.0 are now explicitly set")

if __name__ == '__main__':
    main()