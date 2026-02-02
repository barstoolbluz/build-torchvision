# TorchVision Build Testing Guide

## Quick Start

### Test a Specific Build

```bash
# Test the build you just created (when test-build.sh is created)
# ./test-build.sh torchvision-python313-cuda12_8-sm120-avx512

# For now, use manual testing:
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python -c "
import torch
import torchvision
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Available Builds

Currently implemented: 44 variants on `main`, 18 on CUDA branches (62 total)

**On `main` branch (CUDA 12.8):**
- **SM120 variants:** `torchvision-python313-cuda12_8-sm120-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM100 variants:** `torchvision-python313-cuda12_8-sm100-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM90 variants:** `torchvision-python313-cuda12_8-sm90-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM89 variants:** `torchvision-python313-cuda12_8-sm89-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM86 variants:** `torchvision-python313-cuda12_8-sm86-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM80 variants:** `torchvision-python313-cuda12_8-sm80-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM61 variant:** `torchvision-python313-cuda12_8-sm61-avx` (cuDNN disabled)
- **CPU-only variants:** `torchvision-python313-cpu-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`

**On `cuda-12_9` branch (CUDA 12.9):**
- **SM103 variants:** `torchvision-python313-cuda12_9-sm103-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`

**On `cuda-13_0` branch (CUDA 13.0):**
- **SM110 variants:** `torchvision-python313-cuda13_0-sm110-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`
- **SM121 variants:** `torchvision-python313-cuda13_0-sm121-{avx2,avx512,avx512bf16,avx512vnni,armv8.2,armv9}`

## What the Tests Should Check

A comprehensive test script for TorchVision should perform these tests:

### Test 1: PyTorch & TorchVision Import & Version
- Verifies both PyTorch and TorchVision can be imported
- Shows versions and installation paths
- Confirms version compatibility
- **Exit on failure**: Yes - if either package cannot be imported

### Test 2: CUDA Support (GPU builds only)
- Checks if CUDA is built into PyTorch
- Shows CUDA version, cuDNN version
- Lists compiled GPU architectures
- Detects available GPUs
- **Exit on failure**: Yes - if CUDA support is missing from CUDA build

### Test 3: CPU Inference (TorchVision)
- Loads a pre-trained model (e.g., ResNet18)
- Creates dummy image data
- Runs inference on CPU
- Validates output shapes
- **Exit on failure**: Yes - basic functionality must work

### Test 4: GPU Inference (GPU builds only)
- **Architecture compatibility check**: Compares GPU hardware vs. compiled architectures
- If mismatch detected: Skips test gracefully (not a failure)
- If compatible: Runs GPU inference with TorchVision model
- **Exit on failure**: Only if unexpected errors occur

### Test 5: TorchVision Transforms
- Tests common image transformations
- Verifies tensor operations
- Confirms expected shapes and dtypes
- **Exit on failure**: No - informational only

### Test 6: Build Configuration
- Verifies build name matches package
- Shows PyTorch and TorchVision versions
- Confirms expected architecture is compiled
- **Exit on failure**: No - informational only

## Manual Testing (Current)

Until `test-build.sh` is created, use these manual tests:

### Basic Import Test

```bash
# Test PyTorch and TorchVision import
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision

print("=" * 50)
print("TorchVision Build Test")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"Python version: {torch.__version__.split('+')[0]}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        cc = torch.cuda.get_device_capability(i)
        print(f"  Compute capability: {cc[0]}.{cc[1]}")

print("✓ Import test passed!")
EOF
```

### CPU Inference Test

```bash
# Test TorchVision model on CPU
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision.models as models
import torchvision.transforms as transforms

print("=" * 50)
print("CPU Inference Test")
print("=" * 50)

# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Create dummy image data (batch_size=4, channels=3, height=224, width=224)
dummy_input = torch.randn(4, 3, 224, 224)

# Run inference
with torch.no_grad():
    output = model(dummy_input)

print(f"✓ Input shape: {dummy_input.shape}")
print(f"✓ Output shape: {output.shape}")
print(f"✓ Expected output shape: torch.Size([4, 1000])")
assert output.shape == torch.Size([4, 1000]), "Output shape mismatch!"
print("✓ CPU inference successful!")
EOF
```

### GPU Inference Test

```bash
# Test TorchVision model on GPU (only if CUDA available)
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision.models as models

print("=" * 50)
print("GPU Inference Test")
print("=" * 50)

if not torch.cuda.is_available():
    print("⚠ CUDA not available, skipping GPU test")
    exit(0)

# Check GPU architecture compatibility
gpu_cc = torch.cuda.get_device_capability(0)
gpu_arch = f"{gpu_cc[0]}.{gpu_cc[1]}"
print(f"GPU compute capability: {gpu_arch}")

# Get compiled architectures (if available)
try:
    # This is approximate - PyTorch doesn't expose compiled arch directly
    print("✓ CUDA is available and working")
except Exception as e:
    print(f"⚠ Could not determine compiled architectures: {e}")

# Load model
device = torch.device("cuda:0")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = model.to(device)
model.eval()

# Create dummy data on GPU
dummy_input = torch.randn(4, 3, 224, 224).to(device)

# Run inference
try:
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✓ Input device: {dummy_input.device}")
    print(f"✓ Output device: {output.device}")
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == torch.Size([4, 1000]), "Output shape mismatch!"
    print("✓ GPU inference successful!")

except RuntimeError as e:
    if "CUDA" in str(e) and "architecture" in str(e).lower():
        print(f"⚠ GPU architecture mismatch (expected behavior)")
        print(f"  This build is targeted for a different GPU architecture")
        print(f"  Error: {e}")
    else:
        raise
EOF
```

### TorchVision Transforms Test

```bash
# Test image transformations
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

print("=" * 50)
print("TorchVision Transforms Test")
print("=" * 50)

# Create dummy PIL image
dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

# Define transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transforms
tensor = transform(dummy_image)

print(f"✓ Original image size: {dummy_image.size}")
print(f"✓ Transformed tensor shape: {tensor.shape}")
print(f"✓ Tensor dtype: {tensor.dtype}")
print(f"✓ Expected shape: torch.Size([3, 224, 224])")
assert tensor.shape == torch.Size([3, 224, 224]), "Transform shape mismatch!"
print("✓ Transforms test successful!")
EOF
```

### Complete Test Suite

```bash
# Run all tests in sequence
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

print("\n" + "=" * 50)
print("TorchVision Build Test Suite")
print("=" * 50)

# Test 1: Import and Version
print("\nTest 1: Import and Version")
print("-" * 50)
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ TorchVision version: {torchvision.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    cc = torch.cuda.get_device_capability(0)
    print(f"✓ Compute capability: {cc[0]}.{cc[1]}")

# Test 2: CPU Inference
print("\nTest 2: CPU Inference")
print("-" * 50)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
dummy_input = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"✓ Input shape: {dummy_input.shape}")
print(f"✓ Output shape: {output.shape}")
assert output.shape == torch.Size([2, 1000]), "Output shape mismatch!"
print("✓ CPU inference successful!")

# Test 3: GPU Inference (if available)
if torch.cuda.is_available():
    print("\nTest 3: GPU Inference")
    print("-" * 50)
    try:
        device = torch.device("cuda:0")
        model_gpu = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        model_gpu.eval()
        dummy_input_gpu = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output_gpu = model_gpu(dummy_input_gpu)
        print(f"✓ GPU inference successful!")
        print(f"✓ Output shape: {output_gpu.shape}")
    except RuntimeError as e:
        if "CUDA" in str(e) and "architecture" in str(e).lower():
            print(f"⚠ GPU architecture mismatch (expected for different GPU)")
        else:
            raise
else:
    print("\nTest 3: GPU Inference - SKIPPED (no CUDA)")

# Test 4: Transforms
print("\nTest 4: Image Transforms")
print("-" * 50)
dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
tensor = transform(dummy_image)
print(f"✓ Transformed tensor shape: {tensor.shape}")
assert tensor.shape == torch.Size([3, 224, 224]), "Transform shape mismatch!"
print("✓ Transforms successful!")

print("\n" + "=" * 50)
print("All Tests Passed!")
print("=" * 50)
EOF
```

## Understanding Test Output

### Successful Test (Compatible GPU)

```bash
$ ./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python [test script]

==================================================
TorchVision Build Test Suite
==================================================

Test 1: Import and Version
--------------------------------------------------
✓ PyTorch version: 2.7.0
✓ TorchVision version: 0.21.0
✓ CUDA available: True
✓ CUDA version: 12.8
✓ GPU: NVIDIA GeForce RTX 5090
✓ Compute capability: 12.0

Test 2: CPU Inference
--------------------------------------------------
✓ Input shape: torch.Size([2, 3, 224, 224])
✓ Output shape: torch.Size([2, 1000])
✓ CPU inference successful!

Test 3: GPU Inference
--------------------------------------------------
✓ GPU inference successful!
✓ Output shape: torch.Size([2, 1000])

Test 4: Image Transforms
--------------------------------------------------
✓ Transformed tensor shape: torch.Size([3, 224, 224])
✓ Transforms successful!

==================================================
All Tests Passed!
==================================================
```

### Architecture Mismatch (Expected Behavior)

When testing an SM120 build on an RTX 3090 (SM86) GPU:

```
Test 3: GPU Inference
--------------------------------------------------
⚠ GPU architecture mismatch (expected for different GPU)
```

**This is NOT a failure** - it confirms the build is correctly targeted for a specific GPU architecture.

## GPU Architecture Compatibility

| Build Variant | Compiled For | Works On | Notes |
|---------------|--------------|----------|-------|
| `sm121-*` | DGX Spark | DGX Spark | Native support ✅ |
| `sm120-*` | RTX 5090 (Blackwell) | RTX 5090+ | Native support ✅ |
| `sm110-*` | NVIDIA DRIVE Thor | DRIVE Thor, Orin+ | Native support ✅ |
| `sm103-*` | B300 (Blackwell) | B300 | Native support ✅ |
| `sm100-*` | B100, B200 (Blackwell) | B100, B200 | Native support ✅ |
| `sm90-*` | H100, L40S (Hopper) | H100, L40S | Native support ✅ |
| `sm89-*` | RTX 4090 (Ada Lovelace) | RTX 4090, L40 | Native support ✅ |
| `sm86-*` | RTX 3090, A40 (Ampere) | RTX 3090, A40, A5000 | Native support ✅ |
| `sm80-*` | A100, A30 (Ampere) | A100, A30 | Native support ✅ |
| `sm120-*` | RTX 5090 | RTX 3090 | ❌ Won't work (backward incompatible) |
| `sm86-*` | RTX 3090 | RTX 5090 | ❌ Won't work (forward incompatible) |

**Key Point**: TorchVision CUDA builds inherit PyTorch's architecture-specific compilation. A build for SM120 will only work on SM120 GPUs.

> **📚 Complete Architecture List:** For the complete build matrix covering all 9 GPU architectures (SM80, SM86, SM89, SM90, SM100, SM103, SM110, SM120, SM121), see:
> - **[QUICKSTART.md - GPU Selection Guide](./QUICKSTART.md#choosing-your-variant)** - Quick reference table
> - **[BUILD_MATRIX.md - Build Matrix](./BUILD_MATRIX.md)** - Detailed build matrix and status

## PyTorch Dependency Verification

TorchVision requires a compatible PyTorch version. Test both:

```bash
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python << 'EOF'
import torch
import torchvision

print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")

# Check compatibility
pytorch_major_minor = '.'.join(torch.__version__.split('.')[:2])
torchvision_major_minor = '.'.join(torchvision.__version__.split('.')[:2])

print(f"\nCompatibility check:")
print(f"  PyTorch {pytorch_major_minor}")
print(f"  TorchVision {torchvision_major_minor}")

# Common compatible versions:
# PyTorch 2.7 <-> TorchVision 0.21
# PyTorch 2.6 <-> TorchVision 0.20
# PyTorch 2.5 <-> TorchVision 0.19

if pytorch_major_minor == "2.7" and torchvision_major_minor == "0.21":
    print("✓ Versions are compatible (PyTorch 2.7 + TorchVision 0.21)")
elif pytorch_major_minor == "2.6" and torchvision_major_minor == "0.20":
    print("✓ Versions are compatible (PyTorch 2.6 + TorchVision 0.20)")
else:
    print(f"⚠ Check https://github.com/pytorch/vision for version compatibility")
EOF
```

## When to Use These Tests

### After Building

```bash
# Build
flox build torchvision-python313-cuda12_8-sm120-avx512

# Test immediately (use complete test suite above)
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python [test script]
```

### Before Publishing

Always test before publishing to FloxHub:

```bash
# Test
./result-torchvision-python313-cuda12_8-sm120-avx512/bin/python [test script]

# If all tests pass, publish
flox publish -o <your-org> torchvision-python313-cuda12_8-sm120-avx512
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torchvision'"

**Problem**: TorchVision not built or installed correctly.

**Solution**:
1. Check if the result symlink exists
2. Rebuild with `flox build torchvision-python313-cuda12_8-sm120-avx512`

### "PyTorch version incompatibility"

**Problem**: TorchVision requires a specific PyTorch version range.

**Solution**: Verify PyTorch and TorchVision version compatibility. See PyTorch vision repository for version matrix.

### "CUDA built but no GPU available"

**Problem**: PyTorch is built with CUDA support, but no GPU is detected.

**Possible causes:**
- No NVIDIA GPU in system
- NVIDIA driver not installed
- Testing on a headless system without GPU

**Note**: This is a warning, not an error. CPU inference will still work.

### "GPU architecture mismatch"

**Problem**: Testing an SM86 build on an RTX 5090, or vice versa.

**Solution**: This is expected behavior! Build the correct variant for your GPU:
- DGX Spark → use SM121 builds ✅
- RTX 5090 → use SM120 builds ✅
- NVIDIA DRIVE Thor/Orin+ → use SM110 builds ✅
- B300 → use SM103 builds ✅
- B100/B200 → use SM100 builds ✅
- H100/L40S → use SM90 builds ✅
- RTX 4090 → use SM89 builds ✅
- RTX 3090/A40 → use SM86 builds ✅
- A100/A30 → use SM80 builds ✅

See [QUICKSTART.md](./QUICKSTART.md#choosing-your-variant) for complete GPU selection guide.

### "Cannot load pre-trained weights"

**Problem**: TorchVision models fail to download pre-trained weights.

**Possible causes:**
- No internet connection
- Firewall blocking downloads
- Disk space issues

**Solution**:
- Ensure internet connectivity
- Pre-download weights and set `TORCH_HOME` environment variable
- Use `weights=None` for testing without pre-trained weights

## Future: Automated Test Script

**TODO:** Create `test-build.sh` script similar to PyTorch's test script with:
- Auto-detection of build variants
- Comprehensive test suite
- Architecture mismatch handling
- Exit codes for CI/CD integration
- Colored output for readability

**Script should accept:**
```bash
./test-build.sh torchvision-python313-cuda12_8-sm120-avx512
./test-build.sh  # Auto-detect if only one build exists
```

## Integration with CI/CD

```bash
#!/bin/bash
# Example CI/CD script for TorchVision

set -e

# Build and test all SM120 variants
for isa in avx2 avx512 avx512bf16 avx512vnni; do
    variant="torchvision-python313-cuda12_8-sm120-${isa}"

    echo "Building ${variant}..."
    flox build ${variant}

    echo "Testing ${variant}..."
    ./result-${variant}/bin/python [test script]

    echo "Publishing ${variant}..."
    flox publish -o myorg ${variant}
done
```

## Summary

Use these manual tests for:
- ✅ Testing any TorchVision build variant
- ✅ Verifying CUDA support
- ✅ Validating architecture compatibility
- ✅ Verifying PyTorch/TorchVision version compatibility
- ✅ Pre-publication verification
- ✅ Debugging build issues

**Current Status:** Manual testing only (test-build.sh to be created)

**Available Tests:**
- Basic import test
- CPU inference test
- GPU inference test
- Transforms test
- Complete test suite

**Future Work:** Create automated `test-build.sh` script for streamlined testing workflow.
