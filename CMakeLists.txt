cmake_minimum_required(VERSION 3.20)
project(plastix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLASTIX_BUILD_TESTS "Build tests" ON)
option(PLASTIX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(PLASTIX_BUILD_EXAMPLES "Build examples" ON)
option(PLASTIX_ENABLE_CUDA "Enable CUDA support" OFF)

add_library(plastix)
target_sources(plastix PRIVATE src/plastix.cpp)

target_include_directories(plastix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(plastix PRIVATE -Wall -Wextra -Wpedantic)

# CCCL (Thrust + CUB + libcudacxx) is a hard dependency: random.hpp uses
# thrust::random for host/device-portable RNG, and cuda_primitives.hpp
# wraps cub::DeviceScan / cub::DeviceRadixSort. CCCL ships with the CUDA
# toolkit but its CMake config lives outside the default search path, so
# we use CUDAToolkit's discovery to point find_package at the right dir.
find_package(CUDAToolkit REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${CUDAToolkit_LIBRARY_ROOT}")
find_package(CCCL CONFIG REQUIRED)
target_link_libraries(plastix PUBLIC CCCL::CCCL)

# When PLASTIX_ENABLE_CUDA is on, every .cpp in the project is compiled as
# CUDA so nvcc sees PLASTIX_HD-marked code in both host and device flavors.
# The library itself, the tests, and the examples flip language uniformly.
# This helper centralizes the language-flip + compile-options policy. It
# expects the targets it's given to already exist.
function(plastix_enable_cuda_on_target target)
    if(NOT PLASTIX_ENABLE_CUDA)
        return()
    endif()
    get_target_property(target_sources_list ${target} SOURCES)
    foreach(src ${target_sources_list})
        get_filename_component(src_ext ${src} EXT)
        if(src_ext STREQUAL ".cpp")
            set_source_files_properties(${src} TARGET_DIRECTORY ${target}
                                        PROPERTIES LANGUAGE CUDA)
        endif()
    endforeach()
    # Relax host-flag warnings that are noisy under nvcc's host-pass invocation
    # but not actionable, and silence the "extended-lambda" notice on policies.
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endfunction()

if(PLASTIX_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "PLASTIX_ENABLE_CUDA=ON but no CUDA compiler was found. "
            "Install the CUDA Toolkit (nvcc) or disable PLASTIX_ENABLE_CUDA.")
    endif()
    # Default to a fixed multi-arch list so reconfigures don't depend on
    # the host's GPU being available at configure time. Pass
    # -DCMAKE_CUDA_ARCHITECTURES=native if you want to target only the
    # current device. No 75 because we rely on fast 32-bit float atomics.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
            "CUDA arches to compile for" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    file(GLOB PLASTIX_CUDA_SOURCES src/kernels/*.cu)
    if(PLASTIX_CUDA_SOURCES)
        target_sources(plastix PRIVATE ${PLASTIX_CUDA_SOURCES})
    endif()
    target_compile_definitions(plastix PUBLIC PLASTIX_HAS_CUDA)
    set_target_properties(plastix PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    plastix_enable_cuda_on_target(plastix)
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
