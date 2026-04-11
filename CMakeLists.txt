cmake_minimum_required(VERSION 3.20)
project(plastix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PLASTIX_BUILD_TESTS "Build tests" ON)
option(PLASTIX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(PLASTIX_BUILD_EXAMPLES "Build examples" ON)
option(PLASTIX_ENABLE_CUDA "Enable CUDA support" OFF)

add_library(plastix src/plastix.cpp)

target_include_directories(plastix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(plastix PRIVATE -Wall -Wextra -Wpedantic)

if(PLASTIX_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    endif()
    file(GLOB PLASTIX_CUDA_SOURCES src/kernels/*.cu)
    if(PLASTIX_CUDA_SOURCES)
        target_sources(plastix PRIVATE ${PLASTIX_CUDA_SOURCES})
    endif()
    target_compile_definitions(plastix PUBLIC PLASTIX_HAS_CUDA)
endif()

include(FetchContent)

if(PLASTIX_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    add_subdirectory(tests)
endif()

if(PLASTIX_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(PLASTIX_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )
    FetchContent_MakeAvailable(googlebenchmark)
    add_subdirectory(benchmarks)
endif()
