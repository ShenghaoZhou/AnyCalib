#!/bin/bash
set -e

# AnyCalib C++ Build Script

# 1. Create build directory
mkdir -p build

# 2. Configure with CMake
cd build
cmake ..

# 3. Compile
make -j$(nproc)

echo "Build complete! Use './anycalib_inference <engine_path> <image_path>'"
