cmake_minimum_required(VERSION 3.10)
project(AnyCalibInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(anycalib_inference anycalib_inference.cpp)

# EXPLICITLY force include directories if cmake filters them out
target_include_directories(anycalib_inference PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# If the above is filtered out, this will force it
target_compile_options(anycalib_inference PRIVATE "-I${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(anycalib_inference 
    ${OpenCV_LIBRARIES} 
    nvinfer 
    nvonnxparser 
    CUDA::cudart
)
