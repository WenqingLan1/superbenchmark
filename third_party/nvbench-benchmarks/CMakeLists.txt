cmake_minimum_required(VERSION 3.20.1)
project(NVBenchBenchmarks CUDA)

add_subdirectory(nvbench)

function(nvbench_add_benchmark bench_name bench_src)
  add_executable(${bench_name} ${bench_src})
  target_link_libraries(${bench_name} PRIVATE nvbench::main)
  nvbench_config_target(${bench_name})
endfunction()

nvbench_add_benchmark(nvbench_kernel_launch    launch_bench.cu)
# to add more benchmarks
# nvbench_add_benchmark(nvbench_example_bench  example_bench.cu)