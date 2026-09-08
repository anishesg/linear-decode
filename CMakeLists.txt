cmake_minimum_required(VERSION 3.20)
project(linear_decode LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/src)

# Static library from all kernel sources
add_library(linear_decode_kernels STATIC
    src/reference.cu
    src/outer_product.cuh
    src/state_query.cuh
    src/fused_decode.cu
    src/multi_head.cu
)
set_target_properties(linear_decode_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(linear_decode_kernels PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(linear_decode_kernels CUDA::cublas CUDA::cudart)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness linear_decode_kernels)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency linear_decode_kernels)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CTest integration
enable_testing()
add_test(NAME correctness COMMAND test_correctness)
