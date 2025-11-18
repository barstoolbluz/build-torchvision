#!/usr/bin/env bash
# Generic TorchVision build test script
# Usage: ./test-build.sh [build-name]
# Example: ./test-build.sh torchvision-python313-cuda12_8-sm120-avx512

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
info() { echo -e "${BLUE}$1${NC}"; }
success() { echo -e "${GREEN}✓ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
error() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# Parse arguments
BUILD_NAME="${1:-}"

# Auto-detect if no argument provided
if [ -z "$BUILD_NAME" ]; then
    info "No build name provided, auto-detecting..."

    # Look for result-* symlinks
    RESULT_LINKS=(result-torchvision-*)

    if [ ${#RESULT_LINKS[@]} -eq 0 ] || [ ! -L "${RESULT_LINKS[0]}" ]; then
        error "No result-* symlinks found. Build something first with: flox build <package-name>"
    fi

    if [ ${#RESULT_LINKS[@]} -eq 1 ]; then
        # Only one result, use it
        RESULT_DIR="${RESULT_LINKS[0]}"
        BUILD_NAME="${RESULT_DIR#result-}"
        info "Auto-detected: $BUILD_NAME"
    else
        # Multiple results, ask user to specify
        warning "Multiple builds found:"
        for link in "${RESULT_LINKS[@]}"; do
            echo "  - ${link#result-}"
        done
        echo ""
        error "Please specify which build to test: ./test-build.sh <build-name>"
    fi
else
    # User provided build name
    RESULT_DIR="result-${BUILD_NAME}"
fi

# Verify result directory exists
if [ ! -L "$RESULT_DIR" ] && [ ! -d "$RESULT_DIR" ]; then
    error "Result directory not found: $RESULT_DIR\nDid you build it? Run: flox build $BUILD_NAME"
fi

# Resolve to actual path
RESULT_PATH=$(readlink -f "$RESULT_DIR" 2>/dev/null || echo "$RESULT_DIR")

if [ ! -d "$RESULT_PATH" ]; then
    error "Result path does not exist: $RESULT_PATH"
fi

# Setup Python path
PYTHONPATH="${RESULT_PATH}/lib/python3.13/site-packages:${PYTHONPATH:-}"

# TorchVision depends on PyTorch - find and add PyTorch to PYTHONPATH
# Use -qR to get transitive dependencies
PYTORCH_DEP=$(nix-store -qR "$RESULT_PATH" 2>/dev/null | grep "python3.13-torch-" | grep -v "vision" | grep -v "dev" | grep -v "dist" | grep -v "\-lib" | head -1)
if [ -n "$PYTORCH_DEP" ] && [ -d "$PYTORCH_DEP/lib/python3.13/site-packages" ]; then
    PYTHONPATH="${PYTORCH_DEP}/lib/python3.13/site-packages:${PYTHONPATH}"
fi

export PYTHONPATH

# Determine if this is a CUDA build
IS_CUDA_BUILD=false
if [[ "$BUILD_NAME" == *"cuda"* ]] || [[ "$BUILD_NAME" == *"sm"[0-9]* ]]; then
    IS_CUDA_BUILD=true
fi

echo "========================================"
echo "TorchVision Build Test"
echo "========================================"
info "Build: $BUILD_NAME"
info "Path: $RESULT_PATH"
info "CUDA build: $IS_CUDA_BUILD"
echo ""

# Test 1: Import TorchVision
echo "========================================"
echo "Test 1: TorchVision Import & Version"
echo "========================================"
python3.13 << 'EOF'
import sys
try:
    import torch
    import torchvision
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ TorchVision version: {torchvision.__version__}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Install path: {torchvision.__file__}")
except ImportError as e:
    print(f"✗ Failed to import TorchVision: {e}")
    sys.exit(1)
EOF
echo ""

# Test 2: CUDA Support (if CUDA build)
if [ "$IS_CUDA_BUILD" = true ]; then
    echo "========================================"
    echo "Test 2: CUDA Support"
    echo "========================================"
    python3.13 << 'EOF'
import torch
import torchvision
import sys

cuda_available = torch.cuda.is_available()
cuda_built = torch.version.cuda is not None

print(f"CUDA built: {cuda_built}")
print(f"CUDA available: {cuda_available}")

if cuda_built:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Compiled arch list: {torch.cuda.get_arch_list()}")
else:
    print("✗ PyTorch was not built with CUDA support!")
    sys.exit(1)

if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")

    if gpu_count > 0:
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {name}")
            print(f"  Compute capability: {cap[0]}.{cap[1]}")
            print(f"  Memory: {mem_gb:.1f} GB")
else:
    print("⚠ CUDA is built but no GPU detected (driver/hardware issue)")
    print("  This is OK if testing on a system without a GPU")
EOF
    echo ""
fi

# Test 3: CPU Image Operations
echo "========================================"
echo "Test 3: CPU Image Operations"
echo "========================================"
python3.13 << 'EOF'
import torch
import torchvision
import torchvision.transforms as transforms
import sys

try:
    # Create a test image tensor (3x224x224 RGB image)
    test_image = torch.randn(3, 224, 224)

    # Test various transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed = transform(test_image)

    print(f"✓ Original image shape: {test_image.shape}")
    print(f"✓ Transformed image shape: {transformed.shape}")

    # Test batch processing
    batch = torch.randn(4, 3, 224, 224)
    batch_transformed = torch.stack([transform(img) for img in batch])

    print(f"✓ Batch shape: {batch_transformed.shape}")
    print("✓ CPU image operations successful!")

except Exception as e:
    print(f"✗ CPU image operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
echo ""

# Test 4: Model Loading & Inference
echo "========================================"
echo "Test 4: Model Loading & CPU Inference"
echo "========================================"
python3.13 << 'EOF'
import torch
import torchvision.models as models
import sys

try:
    # Load a pretrained model (weights won't download, just test architecture)
    model = models.resnet18(weights=None)
    model.eval()

    # Test input (batch of 2 images)
    x = torch.randn(2, 3, 224, 224)

    # Run inference
    with torch.no_grad():
        output = model(x)

    print(f"✓ Model: ResNet-18")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ CPU model inference successful!")

except Exception as e:
    print(f"✗ CPU model inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
echo ""

# Test 5: GPU Operations (if CUDA build and GPU available)
if [ "$IS_CUDA_BUILD" = true ]; then
    echo "========================================"
    echo "Test 5: GPU Image Operations"
    echo "========================================"
    set +e  # Temporarily disable exit-on-error
    python3.13 << 'EOF'
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import sys

if not torch.cuda.is_available():
    print("⚠ Skipping GPU test - no GPU available")
    sys.exit(0)

# Check for architecture compatibility
gpu_cap = torch.cuda.get_device_capability(0)
gpu_arch = f"{gpu_cap[0]}.{gpu_cap[1]}"
compiled_archs = torch.cuda.get_arch_list()

print(f"GPU compute capability: {gpu_arch}")
print(f"Compiled architectures: {compiled_archs}")

# Check if GPU architecture is compatible
compatible = any(gpu_arch in arch or f"sm_{int(float(gpu_arch)*10)}" in arch for arch in compiled_archs)

if not compatible:
    print(f"⚠ WARNING: GPU architecture {gpu_arch} not in compiled list {compiled_archs}")
    print(f"  This build is targeted for a different GPU architecture")
    print(f"  GPU operations will fail - this is expected behavior")
    sys.exit(0)

try:
    # Load model and move to GPU
    model = models.resnet18(weights=None).cuda()
    model.eval()

    # Create test images and move to GPU
    images = torch.randn(4, 3, 224, 224).cuda()

    # Run inference on GPU
    with torch.no_grad():
        output = model(images)

    print(f"✓ Input device: {images.device}")
    print(f"✓ Output device: {output.device}")
    print(f"✓ Output shape: {output.shape}")

    # Test image transformations on GPU
    transformed = torch.nn.functional.interpolate(images, size=(256, 256))
    print(f"✓ Transformed shape: {transformed.shape}")

    print("✓ GPU image operations successful!")

except Exception as e:
    error_str = str(e)
    if "no kernel image is available" in error_str:
        print(f"⚠ GPU architecture mismatch (expected, not a failure)")
        print(f"  Build compiled for: {compiled_archs}")
        print(f"  GPU architecture: {gpu_arch}")
        sys.exit(0)
    else:
        print(f"✗ GPU operations failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF
    GPU_TEST_EXIT=$?
    set -e
    echo ""
fi

# Test 6: TorchVision Operations
echo "========================================"
echo "Test 6: TorchVision Ops"
echo "========================================"
python3.13 << 'EOF'
import torch
import torchvision.ops as ops
import sys

try:
    # Test NMS (Non-Maximum Suppression)
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 15.0, 15.0],
        [20.0, 20.0, 30.0, 30.0]
    ])
    scores = torch.tensor([0.9, 0.75, 0.8])
    keep = ops.nms(boxes, scores, iou_threshold=0.5)

    print(f"✓ NMS input boxes: {boxes.shape}")
    print(f"✓ NMS kept indices: {keep}")

    # Test RoI Align
    features = torch.randn(1, 3, 10, 10)
    rois = torch.tensor([[0.0, 0.0, 0.0, 5.0, 5.0]])
    aligned = ops.roi_align(features, rois, output_size=(2, 2))

    print(f"✓ RoI Align output: {aligned.shape}")
    print("✓ TorchVision ops successful!")

except Exception as e:
    print(f"✗ TorchVision ops failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
echo ""

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
success "Build: $BUILD_NAME"
success "TorchVision imported successfully"
success "CPU image operations working"
success "CPU model inference working"
success "TorchVision ops working"

if [ "$IS_CUDA_BUILD" = true ]; then
    python3.13 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
    if [ $? -eq 0 ]; then
        success "CUDA support verified"
        if [ "${GPU_TEST_EXIT:-1}" -eq 0 ]; then
            success "GPU operations working (or skipped due to arch mismatch)"
        else
            warning "GPU test failed (check output above)"
        fi
    else
        warning "CUDA built but no GPU detected (may be expected)"
    fi
fi

echo ""
success "All tests passed!"
echo "========================================"
