# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

# Source files
set(SOURCES
    gpu_stream_test.cpp
    gpu_stream_utils.cpp
    gpu_stream.cu
    gpu_stream_kernels.cu
)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    
    # Override CUDA architectures to support newer CUDA versions
    # Remove deprecated architectures that are not supported in CUDA 13.0+
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
        set(NVCC_ARCHS_SUPPORTED "75;80;86;89;90;100;103")
    endif()
    
    add_executable(gpu_stream ${SOURCES})
    set_property(TARGET gpu_stream PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_include_directories(gpu_stream PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gpu_stream numa nvidia-ml)
else()
    # TODO: test for ROC
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found ROCm: " ${HIP_VERSION})

        # Convert cuda code to hip code in cpp
        execute_process(COMMAND hipify-perl -print-stats -o gpu_stream.cpp ${SOURCES} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

        # link hip device lib
        add_executable(gpu_stream gpu_stream.cpp)

        include(CheckSymbolExists)
        check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)
        if(${HIP_UNCACHED_MEMORY})
            target_compile_definitions(gpu_stream PRIVATE HIP_UNCACHED_MEMORY)
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
        target_link_libraries(gpu_stream numa hip::device)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()

install(TARGETS gpu_stream RUNTIME DESTINATION bin)
