cmake_minimum_required(VERSION 3.20.1)

project(NVBenchBenchmarks CUDA)

add_subdirectory(nvbench)

add_executable(nvbench_kernel_launch launch_bench.cu)
target_link_libraries(nvbench_kernel_launch nvbench::main)