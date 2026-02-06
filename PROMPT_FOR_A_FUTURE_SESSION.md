# Future Session: Create Additional TorchVision Build Recipes

## Objective

Create TorchVision build recipes for GPU arch / ISA combinations that exist in `build-pytorch` but are missing from `build-torchvision`.

## Background

All existing TorchVision variants are already using the working three-overlay pattern:
- Overlay 1: CUDA 13.0 (`cudaPackages = final.cudaPackages_13`)
- Overlay 2: MAGMA patch for CUDA 13.0 (`cuda-13.0-clockrate-fix.patch`)
- Overlay 3: PyTorch 2.10.0 source upgrade

New variants should follow this same pattern.

## Missing Variants to Create

### From build-pytorch (confirmed working)

| Variant | GPU Target | CPU ISA | Platform | Notes |
|---------|------------|---------|----------|-------|
| `sm120-avx` | SM120 (RTX 5090) | AVX | x86_64-linux | Broader x86 compatibility (no AVX-512 required) |

### Potential Future Variants

These may be useful depending on hardware availability:

| Variant | GPU Target | CPU ISA | Platform | Notes |
|---------|------------|---------|----------|-------|
| `sm121-armv9-nightly` | SM121 (DGX Spark) | ARMv9 | aarch64-linux | Matches PyTorch nightly variant (different build approach) |

## Reference Implementation

Use `.flox/pkgs/torchvision-python313-cuda13_0-sm120-avx512.nix` as the canonical reference for x86 variants.

Use `.flox/pkgs/torchvision-python313-cuda13_0-sm121-armv9.nix` as the canonical reference for ARM variants.

### Key Elements (already present in all existing variants)

1. **Nixpkgs pin**: `6a030d535719c5190187c4cec156f335e95e3211`
2. **Three overlays**: CUDA 13.0 → MAGMA patch → PyTorch 2.10.0
3. **customPytorch**: Full PyTorch 2.10.0 with all CUDA 13.0 fixes embedded
4. **TorchVision override**: `torchvision.override { torch = customPytorch }`

## Creating a New Variant

1. Copy the closest existing variant (e.g., `sm120-avx512.nix` for `sm120-avx`)
2. Update `cpuFlags` for the target ISA:
   ```nix
   # For sm120-avx (broader compatibility):
   cpuFlags = [
     "-mavx"        # AVX instructions
     "-mfma"        # Fused multiply-add
     "-mf16c"       # Half-precision conversions
   ];
   ```
3. Update `pname` in both `customPytorch` and TorchVision `overrideAttrs`
4. Update `meta.description` and `meta.longDescription`
5. Update build output echo statements

## Post-Creation Documentation Updates

After creating new variants, update:

1. **`/home/daedalus/dev/builds/build-torchvision/docs/torchvision-0.24-cuda13-build-notes.md`**
   - Add new variants to the "Available Variants" table
   - Update any variant-specific notes

2. **`/home/daedalus/dev/builds/build-torchvision/README.md`**
   - Document new build variants
   - Add build instructions for each new target

## Verification

After creating each new variant:
1. Verify three-overlay pattern: `grep -c 'cudaPackages_13\|cuda-13.0-clockrate-fix.patch\|version = "2.10.0"' <file>` (should be 3)
2. Verify correct nixpkgs pin: `grep '6a030d535719c5190187c4cec156f335e95e3211' <file>`
3. Verify MAGMA enabled in echo: `grep 'MAGMA.*Enabled' <file>`
4. Build test (if hardware available): `flox build <package-name>`

## Alignment with build-pytorch

Ensure TorchVision variants align with their PyTorch counterparts:

| PyTorch Variant | TorchVision Variant | Status |
|-----------------|---------------------|--------|
| sm110-armv8_2 | sm110-armv8_2 | ✓ Exists |
| sm110-armv9 | sm110-armv9 | ✓ Exists |
| sm120-avx | sm120-avx | **Missing** |
| sm120-avx512 | sm120-avx512 | ✓ Exists |
| sm121-armv8_2 | sm121-armv8_2 | ✓ Exists (TorchVision only) |
| sm121-armv9 | sm121-armv9 | ✓ Exists (needs PyTorch counterpart) |
