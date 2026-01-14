#!/usr/bin/env python3
"""
Fix CPU variants to properly pin PyTorch 2.8.0
"""

import glob
import re

def fix_cpu_variant(filepath):
    """Fix a single CPU variant file."""
    print(f"Fixing: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already has pytorch_2_8_0
    if 'pytorch_2_8_0' in content:
        print(f"  Already fixed, skipping")
        return

    # Find where customPytorch is defined and fix it
    pattern = r'(  # Custom PyTorch with matching CPU configuration\n  customPytorch = \(python3Packages\.torch\.overrideAttrs)'

    replacement = '''  # Pin PyTorch to 2.8.0 for compatibility with TorchVision 0.23.0
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
  customPytorch = (pytorch_2_8_0.overrideAttrs'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Remove the extra )) at the end of customPytorch definition if it exists
    content = content.replace('    \'\'\';\n  }));', '    \'\'\';\n  });')

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✓ Fixed")

# Fix all CPU variants
cpu_files = glob.glob('.flox/pkgs/torchvision-python313-cpu-*.nix')
print(f"Found {len(cpu_files)} CPU variant files to fix")

for filepath in sorted(cpu_files):
    fix_cpu_variant(filepath)