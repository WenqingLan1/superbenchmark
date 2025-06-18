cmake_minimum_required(VERSION 3.20)
project(nvbench_kernel_launch LANGUAGES CUDA)

set(SB_MICRO_PATH $ENV{SB_MICRO_PATH}
    CACHE PATH "where nvbench was installed (bin/, lib/, include/)"
)

find_package(nvbench CONFIG REQUIRED
  PATHS ${SB_MICRO_PATH}/lib/cmake/nvbench
  NO_DEFAULT_PATH
)

add_executable(nvbench_kernel_launch launch_bench.cu)
target_compile_features(nvbench_kernel_launch PUBLIC cuda_std_17)
target_link_libraries(nvbench_kernel_launch
  PRIVATE nvbench::nvbench nvbench::main
)

install(TARGETS nvbench_kernel_launch RUNTIME DESTINATION bin)