cmake_minimum_required(VERSION 3.18)
project(sam3_build_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this at your TensorRT install if it's not on the default search path,
# e.g.: cmake -B build -DTENSORRT_ROOT=/usr/local/TensorRT-10.x
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT install (contains include/ and lib/)")

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include
    PATH_SUFFIXES include)

find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib
    PATH_SUFFIXES lib)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin
    HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib
    PATH_SUFFIXES lib)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib
    PATH_SUFFIXES lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR
        "TensorRT not found. Pass -DTENSORRT_ROOT=/path/to/TensorRT (must contain include/NvInfer.h and lib/libnvinfer.so).")
endif()

add_executable(build_engine build_engine.cpp)
target_include_directories(build_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(build_engine PRIVATE
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
)
