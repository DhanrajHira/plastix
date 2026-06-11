cmake_minimum_required(VERSION 4.0)
project(plastix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLASTIX_BUILD_TESTS "Build tests" ON)
option(PLASTIX_BUILD_BENCHMARKS "Build benchmarks" ON)
option(PLASTIX_BUILD_EXAMPLES "Build examples" ON)
option(PLASTIX_ENABLE_CUDA "Enable CUDA support" OFF)

include(GNUInstallDirs)

# Default install rules ON only when plastix is the top-level project
option(PLASTIX_INSTALL "Generate install + find_package rules for plastix" ${PROJECT_IS_TOP_LEVEL})

add_library(plastix)
add_library(plastix::plastix ALIAS plastix) # common namespace
target_sources(plastix PRIVATE src/plastix.cpp)

# Public headers as a FILE_SET
file(GLOB_RECURSE PLASTIX_PUBLIC_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/plastix/*.hpp")
if(NOT PLASTIX_ENABLE_CUDA)
    list(FILTER PLASTIX_PUBLIC_HEADERS EXCLUDE
         REGEX "/cuda_(kernels|primitives)\\.hpp$")
endif()
target_sources(plastix PUBLIC
    FILE_SET plastix_headers TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES ${PLASTIX_PUBLIC_HEADERS}
)

# Propagate the C++20 requirement to consumers
target_compile_features(plastix PUBLIC cxx_std_20)

target_compile_options(plastix PRIVATE -Wall -Wextra -Wpedantic)

# When PLASTIX_ENABLE_CUDA is on, every .hpp in the project is compiled as CUDA
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
    find_package(CUDAToolkit REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH "${CUDAToolkit_LIBRARY_ROOT}")
    find_package(CCCL CONFIG REQUIRED)
    target_link_libraries(plastix PUBLIC CCCL::CCCL)

    include(CheckLanguage)
    check_language(CUDA)
    set(PROPERTY CUDA_STANDARD_REQUIRED 20)
    find_program(HAS_NVCC nvcc)
    if(NOT HAS_NVCC)
        message(FATAL_ERROR
            "PLASTIX_ENABLE_CUDA=ON but no CUDA compiler was found. "
            "Install the CUDA Toolkit (nvcc) or disable PLASTIX_ENABLE_CUDA.")
    endif()
    # No 75, require 32-bit float atomics.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
            "CUDA arches to compile for" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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

# ---------------------------------------------------------------------------
# Install + find_package(plastix) support
# ---------------------------------------------------------------------------
if(PLASTIX_INSTALL)
    include(CMakePackageConfigHelpers)

    set(PLASTIX_CMAKE_CONFIG_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/plastix")

    install(TARGETS plastix
        EXPORT plastixTargets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        FILE_SET plastix_headers DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    install(EXPORT plastixTargets
        FILE plastixTargets.cmake
        NAMESPACE plastix::
        DESTINATION "${PLASTIX_CMAKE_CONFIG_DESTINATION}"
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/plastixConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/plastixConfig.cmake"
        INSTALL_DESTINATION "${PLASTIX_CMAKE_CONFIG_DESTINATION}"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/plastixConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/plastixConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/plastixConfigVersion.cmake"
        DESTINATION "${PLASTIX_CMAKE_CONFIG_DESTINATION}"
    )
endif()
