#ifndef PLASTIX_CUDA_PRIMITIVES_HPP
#define PLASTIX_CUDA_PRIMITIVES_HPP

// Thin wrappers around CCCL parallel primitives plus a couple of launch
// helpers. The framework used to ship hand-rolled scan and radix-sort
// implementations here; both are now thin adapters over `cub::DeviceScan`
// and `cub::DeviceRadixSort` so the heavy lifting lives in CCCL.

#ifdef PLASTIX_HAS_CUDA

#include "plastix/macros.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace plastix {
namespace cuda {

// Standard threads-per-block for the bulk of the framework's per-element
// kernels. Picked to keep occupancy high on Ampere/Ada without leaving too
// many idle lanes when the work item count isn't a multiple of the warp.
inline constexpr int DefaultBlockSize = 256;

// Grid sizing helper: ceil(N / Block).
PLASTIX_HOST size_t GridSize(size_t N, int Block = DefaultBlockSize) {
  return (N + static_cast<size_t>(Block) - 1) / static_cast<size_t>(Block);
}

// ---------------------------------------------------------------------------
// In-place exclusive scan (CUB)
// ---------------------------------------------------------------------------
//
// `cub::DeviceScan::ExclusiveSum` supports `d_in == d_out`, so we forward
// the caller's pointer to itself. Two-call dance: the first invocation
// only sizes the temp-storage requirement, the second runs the scan.

template <typename T>
PLASTIX_HOST void ExclusiveScanInPlace(T *Data, size_t N) {
  if (N == 0)
    return;
  size_t TempBytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, TempBytes, Data, Data,
                                static_cast<int>(N));
  void *Temp = nullptr;
  cudaMalloc(&Temp, TempBytes);
  cub::DeviceScan::ExclusiveSum(Temp, TempBytes, Data, Data,
                                static_cast<int>(N));
  cudaFree(Temp);
}

// ---------------------------------------------------------------------------
// Test helpers — exercise the launch + atomicAdd<float> path so unit tests
// can verify the toolchain links and runs.
// ---------------------------------------------------------------------------

namespace detail {

// Templated so that header-only inclusion across multiple TUs doesn't
// produce duplicate symbols at device-link time (non-template __global__
// functions get external linkage per TU).
template <typename T = void>
__global__ void AddOnesKernel(float *Slot, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I < N)
    atomicAdd(Slot, 1.0f);
}

} // namespace detail

PLASTIX_HOST void AddOnesIntoSlot(float *Slot, size_t N) {
  if (N == 0)
    return;
  unsigned Grid = static_cast<unsigned>(GridSize(N));
  detail::AddOnesKernel<><<<Grid, DefaultBlockSize>>>(Slot, N);
}

// ---------------------------------------------------------------------------
// In-place 64-bit ascending sort (CUB radix)
// ---------------------------------------------------------------------------
//
// `cub::DeviceRadixSort::SortKeys` is out-of-place and needs a separate
// output buffer plus a temp-storage allocation. We allocate both, run the
// sort, then memcpy the sorted keys back into `Data`. Managed memory keeps
// the scratch buffer addressable from host code (matches the rest of the
// framework's allocation discipline).

PLASTIX_HOST void RadixSort64InPlace(uint64_t *Data, size_t N) {
  if (N <= 1)
    return;
  uint64_t *Scratch = nullptr;
  cudaMallocManaged(&Scratch, N * sizeof(uint64_t));

  size_t TempBytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, TempBytes, Data, Scratch,
                                 static_cast<int>(N));
  void *Temp = nullptr;
  cudaMalloc(&Temp, TempBytes);
  cub::DeviceRadixSort::SortKeys(Temp, TempBytes, Data, Scratch,
                                 static_cast<int>(N));
  cudaDeviceSynchronize();
  cudaMemcpy(Data, Scratch, N * sizeof(uint64_t), cudaMemcpyDefault);

  cudaFree(Temp);
  cudaFree(Scratch);
}

} // namespace cuda
} // namespace plastix

#endif // PLASTIX_HAS_CUDA

#endif // PLASTIX_CUDA_PRIMITIVES_HPP
