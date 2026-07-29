cmake_minimum_required(VERSION 3.30.4)

project(dectris_holoscan_pipeline LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(holoscan REQUIRED CONFIG PATHS /opt/nvidia/holoscan)
find_package(pybind11 CONFIG REQUIRED)

find_library(NVCOMP_LIBRARY
  NAMES nvcomp
  PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu/nvcomp/12
    /usr/lib/sbsa-linux-gnu/nvcomp/12
    /usr/local/lib
  REQUIRED
)
find_path(NVCOMP_INCLUDE_DIR
  NAMES nvcomp.h nvcomp/nvcomp.h
  PATHS
    /usr/include
    /usr/include/nvcomp_13
    /usr/include/nvcomp_12
    /usr/local/include
  REQUIRED
)

find_path(ZMQ_INCLUDE_DIR NAMES zmq.h REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq REQUIRED)

set(DECTRIS_HOLOSCAN_COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common)
set(DECTRIS_HOLOSCAN_OPERATORS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/operators)

add_library(dectris_holoscan_ops_native STATIC
  cpp/common/fast_cbor_parse.c
  cpp/common/bitshuffle_kernel_gpu.cu
  cpp/operators/zmq_rx_operator.cu
  cpp/operators/decompress_gpu_operator.cu
)
set_target_properties(dectris_holoscan_ops_native PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(dectris_holoscan_ops_native
  PUBLIC
    ${DECTRIS_HOLOSCAN_COMMON_INCLUDE_DIR}
    ${DECTRIS_HOLOSCAN_OPERATORS_INCLUDE_DIR}
    ${NVCOMP_INCLUDE_DIR}
    ${ZMQ_INCLUDE_DIR}
)
target_compile_options(dectris_holoscan_ops_native
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_link_libraries(dectris_holoscan_ops_native
  PUBLIC
    holoscan::core
    holoscan::matx
    CUDA::cudart
    ${NVCOMP_LIBRARY}
    ${ZMQ_LIBRARY}
)

pybind11_add_module(dectris_holoscan_ops
  cpp/bindings/dectris_holoscan_ops.cpp
)
target_include_directories(dectris_holoscan_ops
  PRIVATE
    ${DECTRIS_HOLOSCAN_COMMON_INCLUDE_DIR}
    ${DECTRIS_HOLOSCAN_OPERATORS_INCLUDE_DIR}
)
target_link_libraries(dectris_holoscan_ops
  PRIVATE
    dectris_holoscan_ops_native
    holoscan::core
    holoscan::pybind11
)

set(PYTHON_BUILD_DIR ${CMAKE_BINARY_DIR}/python)
set_target_properties(dectris_holoscan_ops PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BUILD_DIR}
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_custom_target(copy_dectris_holoscan_python ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/python/dectris_holoscan
          ${PYTHON_BUILD_DIR}/dectris_holoscan
  COMMENT "Copying Python app package into build/python"
)
