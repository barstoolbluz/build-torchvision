# Wrapper for Flox compatibility
{ pkgs ? import <nixpkgs> {} }:

let
  # Import the actual package definition
  packageDef = import ./torchvision-python313-cuda12_8-sm120-avx512.nix;
in
  # Call it with nixpkgs components
  packageDef {
    python3Packages = pkgs.python3Packages;
    lib = pkgs.lib;
    config = pkgs.config or {};
    cudaPackages = pkgs.cudaPackages;
    addDriverRunpath = pkgs.addDriverRunpath;
    fetchurl = pkgs.fetchurl;
  }