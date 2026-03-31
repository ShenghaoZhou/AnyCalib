cmake_minimum_required(VERSION 3.10)
project(AnyCalibInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)
find_package(CUDA REQUIRED)

# TensorRT location (installed via pixi)
set(TRT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.pixi/envs/default/include)
set(TRT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.pixi/envs/default/lib)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
link_directories(${TRT_LIB_DIR})

add_executable(anycalib_inference anycalib_inference.cpp)
target_link_libraries(anycalib_inference ${OpenCV_LIBRARIES} nvinfer nvonnxparser cudart)
