add_executable(plastix_benchmarks bench_plastix.cpp)
target_link_libraries(plastix_benchmarks PRIVATE plastix benchmark::benchmark_main)

# ---------------------------------------------------------------------------
# Shared microbench output directory. Every bench in this directory writes
# CSV / JSON artifacts under build/benchmarks/_outputs/. The path is baked
# in via target_compile_definitions on each microbench so a bare invocation
# like `./bench_phase_isolated` lands the file in a predictable place.
# ---------------------------------------------------------------------------

set(PLASTIX_BENCH_OUTPUT_DIR "${CMAKE_BINARY_DIR}/benchmarks/_outputs"
    CACHE PATH "Directory where microbench CSVs / JSONs are written")
file(MAKE_DIRECTORY "${PLASTIX_BENCH_OUTPUT_DIR}")

function(plastix_bench_target tgt)
    target_include_directories(${tgt} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${tgt} PRIVATE
        PLASTIX_BENCH_OUTPUT_DIR="${PLASTIX_BENCH_OUTPUT_DIR}")
endfunction()

# ---------------------------------------------------------------------------
# SpMV / GEMV crossover benchmark — Plastix vs Eigen / OpenBLAS / hand-rolled.
# Pulled in via system packages (Arch: eigen + openblas).
# ---------------------------------------------------------------------------

find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 QUIET MODULE)
endif()

set(BLA_VENDOR OpenBLAS)
find_package(BLAS QUIET)

if(TARGET Eigen3::Eigen AND BLAS_FOUND)
    add_executable(bench_spmv_crossover bench_spmv_crossover.cpp)
    target_link_libraries(bench_spmv_crossover
        PRIVATE plastix benchmark::benchmark Eigen3::Eigen ${BLAS_LIBRARIES})
    target_compile_options(bench_spmv_crossover PRIVATE
        -O3 -march=native -ffast-math)
    plastix_bench_target(bench_spmv_crossover)
    # OpenBLAS header lives in /usr/include or /usr/include/openblas depending
    # on the distro; both are on the default search path on Arch but be
    # defensive in case a future packager moves it.
    find_path(OPENBLAS_CBLAS_HEADER cblas.h
        HINTS /usr/include /usr/include/openblas /usr/local/include)
    if(OPENBLAS_CBLAS_HEADER)
        target_include_directories(bench_spmv_crossover PRIVATE
            ${OPENBLAS_CBLAS_HEADER})
    endif()
else()
    message(STATUS
        "bench_spmv_crossover: skipped (Eigen3=${Eigen3_FOUND}, BLAS=${BLAS_FOUND})")
endif()

# ---------------------------------------------------------------------------
# GPU counterpart: Plastix CUDA forward path vs cuBLAS sgemv vs cuSPARSE SpMV.
# Only built when PLASTIX_ENABLE_CUDA=ON; not runnable on machines without a
# CUDA driver, but compiles so the source stays in sync with the CPU bench.
# ---------------------------------------------------------------------------

if(PLASTIX_ENABLE_CUDA)
    add_executable(bench_spmv_crossover_gpu bench_spmv_crossover_gpu.cu)
    target_link_libraries(bench_spmv_crossover_gpu
        PRIVATE plastix benchmark::benchmark
        CUDA::cublas CUDA::cusparse CUDA::cudart)
    set_target_properties(bench_spmv_crossover_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    target_compile_options(bench_spmv_crossover_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    plastix_bench_target(bench_spmv_crossover_gpu)
endif()

# ---------------------------------------------------------------------------
# Phase-isolated bench. Drills into every per-step Plastix phase
# (Forward, Loss, Backward, UpdateConn, PruneConn, CompactConn, AddConn,
# Reset). Lets the per-phase splits in traditional/_results/runs.csv be
# reproduced in isolation.
# ---------------------------------------------------------------------------

add_executable(bench_phase_isolated bench_phase_isolated.cpp)
target_link_libraries(bench_phase_isolated PRIVATE plastix benchmark::benchmark)
target_compile_options(bench_phase_isolated PRIVATE -O3 -march=native)
plastix_bench_target(bench_phase_isolated)

# ---------------------------------------------------------------------------
# Resort-cost bench. Isolates the topology resort that runs whenever
# DoAddConnections / DoCompactConnections sets the NeedsResort flag.
# ---------------------------------------------------------------------------

add_executable(bench_resort_cost bench_resort_cost.cpp)
target_link_libraries(bench_resort_cost PRIVATE plastix benchmark::benchmark)
target_compile_options(bench_resort_cost PRIVATE -O3 -march=native)
plastix_bench_target(bench_resort_cost)

# ---------------------------------------------------------------------------
# Memory-footprint bench. Measures Plastix RSS at construction / steady
# state and (optionally, when libtorch is available) compares against an
# equivalent dense torch::Tensor allocation.
# ---------------------------------------------------------------------------

add_executable(bench_memory_footprint bench_memory_footprint.cpp)
target_link_libraries(bench_memory_footprint PRIVATE plastix benchmark::benchmark)
target_compile_options(bench_memory_footprint PRIVATE -O3 -march=native)
plastix_bench_target(bench_memory_footprint)

# ---------------------------------------------------------------------------
# Time-to-quality bench. Tracks accuracy / loss vs wall-clock on four of
# the traditional/ tasks (static / idempotent / continuous-small /
# continuous-large) using Plastix and (optionally) libtorch.
# ---------------------------------------------------------------------------

add_executable(bench_time_to_quality bench_time_to_quality.cpp)
target_link_libraries(bench_time_to_quality PRIVATE plastix benchmark::benchmark)
target_compile_options(bench_time_to_quality PRIVATE -O3 -march=native)
plastix_bench_target(bench_time_to_quality)

# ---------------------------------------------------------------------------
# LibTorch (CPU). Pulled via FetchContent — the prebuilt CXX11-ABI release
# zip is ~200 MB and only downloaded the first time CMake configures. Set
# PLASTIX_BENCH_FETCH_LIBTORCH=OFF to skip every torch-dependent bench.
# Set PLASTIX_LIBTORCH_URL to point at a local copy of the zip if the
# CDN is unreachable from the build machine.
# ---------------------------------------------------------------------------

option(PLASTIX_BENCH_FETCH_LIBTORCH
    "Download libtorch (CPU) via FetchContent for the torch microbenches" ON)

set(PLASTIX_LIBTORCH_URL
    "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
    CACHE STRING "URL of the libtorch CPU CXX11-ABI zip")

set(PLASTIX_TORCH_AVAILABLE OFF)
if(PLASTIX_BENCH_FETCH_LIBTORCH)
    include(FetchContent)
    FetchContent_Declare(libtorch
        URL ${PLASTIX_LIBTORCH_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(libtorch)
    # The zip extracts into <build>/_deps/libtorch-src/share/cmake/Torch
    set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch"
        CACHE PATH "" FORCE)
    find_package(Torch CONFIG QUIET)
    if(Torch_FOUND)
        set(PLASTIX_TORCH_AVAILABLE ON)
        message(STATUS "LibTorch ready: ${Torch_DIR}")
    else()
        message(WARNING
            "LibTorch downloaded but find_package(Torch) failed; "
            "torch microbenches will be skipped")
    endif()
else()
    message(STATUS
        "PLASTIX_BENCH_FETCH_LIBTORCH=OFF; torch microbenches skipped")
endif()

# ---------------------------------------------------------------------------
# Torch crossover bench. Same shape/density sweep as bench_spmv_crossover so
# the two outputs join on (In, Out, density_bps).
# ---------------------------------------------------------------------------

if(PLASTIX_TORCH_AVAILABLE)
    add_executable(bench_spmv_crossover_torch bench_spmv_crossover_torch.cpp)
    target_link_libraries(bench_spmv_crossover_torch
        PRIVATE plastix benchmark::benchmark ${TORCH_LIBRARIES})
    # Torch's CMake config exports TORCH_CXX_FLAGS for the libstdc++ ABI
    # selector and a few other settings that must reach every TU linking
    # libtorch.
    target_compile_options(bench_spmv_crossover_torch PRIVATE
        -O3 -march=native ${TORCH_CXX_FLAGS})
    plastix_bench_target(bench_spmv_crossover_torch)

    # Wire libtorch into the two benches that compare against PyTorch when
    # available. They compile and run without it (gated behind
    # PLASTIX_HAVE_TORCH at preprocessor time), torch-aware paths only
    # activate when the macro is defined.
    target_link_libraries(bench_memory_footprint PRIVATE ${TORCH_LIBRARIES})
    target_compile_options(bench_memory_footprint PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_definitions(bench_memory_footprint PRIVATE PLASTIX_HAVE_TORCH=1)

    target_link_libraries(bench_time_to_quality PRIVATE ${TORCH_LIBRARIES})
    target_compile_options(bench_time_to_quality PRIVATE ${TORCH_CXX_FLAGS})
    target_compile_definitions(bench_time_to_quality PRIVATE PLASTIX_HAVE_TORCH=1)
endif()

# ---------------------------------------------------------------------------
# CUDA: every in-tree benchmark includes the Plastix headers, which under
# PLASTIX_ENABLE_CUDA pull in CUB/Thrust device code and the <<<>>> dispatch
# path. Those only compile through nvcc, so the benchmark .cpp TUs must be
# built as CUDA, exactly like the examples and tests. (bench_spmv_crossover_gpu
# is already a .cu and is handled at its own definition above.) The host-only
# optimization / libtorch flags set above are forwarded to the host pass via
# nvcc's -forward-unknown-to-host-compiler, so they keep working unchanged.
# ---------------------------------------------------------------------------
if(PLASTIX_ENABLE_CUDA)
    foreach(plastix_bench_cuda_tgt
            plastix_benchmarks
            bench_spmv_crossover
            bench_phase_isolated
            bench_resort_cost
            bench_memory_footprint
            bench_time_to_quality
            bench_spmv_crossover_torch)
        if(TARGET ${plastix_bench_cuda_tgt})
            plastix_enable_cuda_on_target(${plastix_bench_cuda_tgt})
        endif()
    endforeach()
endif()
