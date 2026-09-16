#ifndef PLASTIX_CUDA_PRIMITIVES_HPP
#define PLASTIX_CUDA_PRIMITIVES_HPP

// Thin wrappers over CCCL primitives plus launch helpers. Scan delegates to
// cub::DeviceScan. Radix sort stays hand-rolled: CUB's sort is
// block-cooperative; we want a one-thread-per-key pass.

#ifdef PLASTIX_HAS_CUDA

#include "plastix/macros.hpp"

#include <cub/device/device_scan.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace plastix {
namespace cuda {

// Default threads-per-block for per-element kernels; tuned for occupancy on
// Ampere/Ada.
inline constexpr int DefaultBlockSize = 256;

// Grid sizing helper: ceil(N / Block).
PLASTIX_HOST size_t GridSize(size_t N, int Block = DefaultBlockSize) {
  return (N + static_cast<size_t>(Block) - 1) / static_cast<size_t>(Block);
}

// In-place exclusive scan (CUB): ExclusiveSum supports d_in==d_out, so alias
// the pointer. Two-call dance: first sizes temp storage, second runs the scan.

template <typename T>
PLASTIX_HOST void ExclusiveScanInPlace(T *Data, size_t N) {
  if (N == 0)
    return;
  size_t TempBytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, TempBytes, Data, Data,
                                static_cast<int>(N));
  void *Temp = nullptr;
  PLASTIX_CUDA_CHECK(cudaMalloc(&Temp, TempBytes));
  cub::DeviceScan::ExclusiveSum(Temp, TempBytes, Data, Data,
                                static_cast<int>(N));
  PLASTIX_CUDA_CHECK(cudaFree(Temp));
}

// Test helpers -- exercise launch + atomicAdd<float> so tests verify the
// toolchain links.

namespace detail {

// Templated to avoid duplicate device symbols across TUs (non-template
// __global__ has external linkage).
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
  PLASTIX_CUDA_CHECK_KERNEL();
}

// 64-bit LSB stable radix sort: 64 passes, 1 bit each, scan-based stable
// partition. 1-bit (not byte-radix atomicAdd) stays stable; thread-per-key
// beats block-cooperative CUB at our ~10^4 keys.

namespace detail {

// Flags[I]=1 if bit Bit of Keys[I] is 0. I==N is the scan sentinel (0), so
// scanned Flags[N] holds the total zero-count.
template <typename T = void>
__global__ void RadixBitFlagKernel(size_t N, const uint64_t *Keys,
                                   uint32_t *Flags, int Bit) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I > N)
    return;
  uint32_t V = 0u;
  if (I < N)
    V = ((Keys[I] >> Bit) & 1ull) ? 0u : 1u;
  Flags[I] = V;
}

// Stable scatter: zero-bit keys land at Scan[I]; one-bit keys land in the
// second half at ZeroCount + (I - Scan[I]).
template <typename T = void>
__global__ void RadixBitScatterKernel(size_t N, const uint64_t *In,
                                      uint64_t *Out, const uint32_t *Scan,
                                      uint32_t ZeroCount, int Bit) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  uint64_t K = In[I];
  uint32_t IsZero = ((K >> Bit) & 1ull) ? 0u : 1u;
  uint32_t Zeros = Scan[I];
  uint32_t Pos =
      IsZero ? Zeros : (ZeroCount + static_cast<uint32_t>(I) - Zeros);
  Out[Pos] = K;
}

} // namespace detail

// Sorts Data[0..N) ascending in place (allocates ~N*8+N*4 managed bytes).
// 64 passes (even), so the sorted result lands back in Data.
PLASTIX_HOST void RadixSort64InPlace(uint64_t *Data, size_t N) {
  if (N <= 1)
    return;
  uint64_t *Scratch = nullptr;
  uint32_t *Flags = nullptr;
  PLASTIX_CUDA_CHECK(cudaMallocManaged(&Scratch, N * sizeof(uint64_t)));
  PLASTIX_CUDA_CHECK(cudaMallocManaged(&Flags, (N + 1) * sizeof(uint32_t)));

  uint64_t *In = Data;
  uint64_t *Out = Scratch;
  unsigned Grid = static_cast<unsigned>(GridSize(N));
  unsigned GridP1 = static_cast<unsigned>(GridSize(N + 1));
  for (int Bit = 0; Bit < 64; ++Bit) {
    detail::RadixBitFlagKernel<>
        <<<GridP1, DefaultBlockSize>>>(N, In, Flags, Bit);
    PLASTIX_CUDA_CHECK_KERNEL();
    ExclusiveScanInPlace(Flags, N + 1);
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t ZeroCount = Flags[N];
    detail::RadixBitScatterKernel<>
        <<<Grid, DefaultBlockSize>>>(N, In, Out, Flags, ZeroCount, Bit);
    PLASTIX_CUDA_CHECK_KERNEL();
    uint64_t *Tmp = In;
    In = Out;
    Out = Tmp;
  }
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  PLASTIX_CUDA_CHECK(cudaFree(Scratch));
  PLASTIX_CUDA_CHECK(cudaFree(Flags));
}

} // namespace cuda
} // namespace plastix

#endif // PLASTIX_HAS_CUDA

#endif // PLASTIX_CUDA_PRIMITIVES_HPP
