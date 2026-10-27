cmake_minimum_required(VERSION 4.0)
project(dectris_holoscan_pipeline LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(holoscan 5.0.0 EXACT REQUIRED CONFIG)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.SABIModule)

# nvCOMP is an explicit dependency in SDK 5; the development image no longer ships it.
find_package(nvcomp 5 REQUIRED CONFIG
  PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake/libnvcomp5-dev-cuda-13/nvcomp)
# Debian's development symlink and runtime SONAME live in separate directories.
file(REAL_PATH "${nvcomp_LIBRARY_DIR}/libnvcomp.so" NVCOMP_RUNTIME_LIBRARY)
get_filename_component(NVCOMP_RUNTIME_DIR "${NVCOMP_RUNTIME_LIBRARY}" DIRECTORY)
find_package(HDF5 REQUIRED COMPONENTS C)
find_path(ZMQ_INCLUDE_DIR NAMES zmq.h REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq REQUIRED)
set(IMXRX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(IMXRX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../dectris/native ${CMAKE_BINARY_DIR}/imxrx)

set(PYTHON_BUILD_DIR ${CMAKE_BINARY_DIR}/python)
foreach(schema IN ITEMS compressed_batch decoded_batch)
  # START metadata is at most 16 KiB; image IDs need at most 8 KiB in a separate message.
  # Keep the decoded descriptor envelope below the channel's inline payload bound.
  if(schema STREQUAL "decoded_batch")
    set(SCHEMA_METADATA_CAPACITY 24576)
  else()
    set(SCHEMA_METADATA_CAPACITY 32768)
  endif()
  holoscan_add_flatbuffer_schema(
    TARGET ${schema}_schema SCHEMA schemas/${schema}.fbs
    MAX_SERIALIZED_SIZE ${SCHEMA_METADATA_CAPACITY} PYTHON
    PYTHON_OUTPUT_DIR ${PYTHON_BUILD_DIR}/live4d/schemas)
endforeach()
add_library(dectris_holoscan_ops SHARED
  cpp/bindings/dectris_holoscan_ops.cpp
  cpp/common/fast_cbor_parse.c cpp/common/batch_window.cpp cpp/common/bitshuffle_kernel_gpu.cu
  cpp/common/cbor_value.cpp cpp/common/nxmx_writer.cpp
  cpp/operators/zmq_rx_operator.cu cpp/operators/decompress_gpu_operator.cu
  cpp/operators/rdma_rx_operator.cu cpp/operators/rdma_rings_rx_operator.cu
  cpp/operators/h5_writer_operator.cu)
set_target_properties(dectris_holoscan_ops PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${NVCOMP_RUNTIME_DIR}"
  LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BUILD_DIR}/live4d)
target_include_directories(dectris_holoscan_ops PRIVATE
  cpp/common cpp/operators ${ZMQ_INCLUDE_DIR} ${HDF5_C_INCLUDE_DIRS})
target_compile_options(dectris_holoscan_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_definitions(dectris_holoscan_ops PRIVATE ENABLE_STATUS_VERIFICATION)
target_link_libraries(dectris_holoscan_ops PRIVATE holoscan::core holoscan::logger
  CUDA::cudart nvcomp::nvcomp ${ZMQ_LIBRARY} ${HDF5_C_LIBRARIES} imxrx compressed_batch_schema
  decoded_batch_schema)

add_custom_target(copy_live4d_python ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/python/live4d ${PYTHON_BUILD_DIR}/live4d)
