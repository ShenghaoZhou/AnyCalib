#!/bin/bash
set -e

# AnyCalib C++ Build Script

# 1. Create build directory
mkdir -p build
cd build

if command -v pixi &> /dev/null; then
  echo "Using pixi to build..."
  pixi run cmake ..
  pixi run make -j$(nproc)
else
  # 3. Configure with CMake
  cmake ..
  # 4. Compile
  make -j$(nproc)
fi

# Copy binary to current directory
cp anycalib_inference ..

echo "Build complete! Use './anycalib_inference <engine_path> <image_path>'"
