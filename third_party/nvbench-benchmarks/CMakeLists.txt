cmake_minimum_required(VERSION 3.20.1)

project(NVBenchBenchmarks CUDA)

add_subdirectory(nvbench)

add_executable(launch_bench launch_bench.cu)
target_link_libraries(launch_bench nvbench::main)