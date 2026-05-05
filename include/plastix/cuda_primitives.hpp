#ifndef PLASTIX_CUDA_PRIMITIVES_HPP
#define PLASTIX_CUDA_PRIMITIVES_HPP

// Thin wrappers around CCCL parallel primitives plus a couple of launch
// helpers, alongside an in-house thread-level radix sort that we keep
// hand-rolled on purpose. The exclusive scan delegates to
// `cub::DeviceScan::ExclusiveSum`; the radix sort stays bespoke because
// CUB's device-level sort is built around block-level cooperation, while
// the framework's proposal pipeline wants a cheaper one-thread-per-key
// pass that interleaves cleanly with the scan-based scatter below.

#ifdef PLASTIX_HAS_CUDA

#include "plastix/macros.hpp"

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
// 64-bit LSB stable radix sort (1-bit per pass, scan-based scatter)
// ---------------------------------------------------------------------------
//
// 64 passes, one per bit. Each pass partitions keys into a "0-bit" half
// followed by a "1-bit" half, preserving relative order — a stable parallel
// partition built on top of a single exclusive scan.
//
// Why 1-bit and not 8-bit byte radix: a byte-radix scatter via atomicAdd is
// fast but unstable across keys with the same digit, which corrupts the
// ordering established by lower-bit passes. The scan-based 1-bit partition
// is naturally stable and only marginally more expensive at our N (proposal
// counts are bounded by ConnAlloc capacity, ~10^4 keys).
//
// Why not cub::DeviceRadixSort: that primitive is block-cooperative and
// pulls in a sizable temp-storage / dispatch dance. The pass below is a
// straight thread-level partition (one thread per key per bit) that
// composes naturally with `ExclusiveScanInPlace` above and matches the
// granularity the proposal pipeline expects.
//
// Templated to dodge nvlink ODR violations from non-template __global__
// definitions in a header.

namespace detail {

// Writes Flags[I] = 1 if bit `Bit` of Keys[I] is 0, else 0. Index I == N is
// reserved as the scan sentinel and gets a 0 so the exclusive-scanned value
// at Flags[N] holds the total zero-count.
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
  uint32_t Pos = IsZero ? Zeros : (ZeroCount + static_cast<uint32_t>(I) - Zeros);
  Out[Pos] = K;
}

} // namespace detail

// Sorts `Data[0..N)` ascending in place. Allocates ~N*8 + N*4 bytes of
// managed memory for the duration of the call. 64 passes (even), so the
// final sorted result is guaranteed to be back in `Data`.
PLASTIX_HOST void RadixSort64InPlace(uint64_t *Data, size_t N) {
  if (N <= 1)
    return;
  uint64_t *Scratch = nullptr;
  uint32_t *Flags = nullptr;
  cudaMallocManaged(&Scratch, N * sizeof(uint64_t));
  cudaMallocManaged(&Flags, (N + 1) * sizeof(uint32_t));

  uint64_t *In = Data;
  uint64_t *Out = Scratch;
  unsigned Grid = static_cast<unsigned>(GridSize(N));
  unsigned GridP1 = static_cast<unsigned>(GridSize(N + 1));
  for (int Bit = 0; Bit < 64; ++Bit) {
    detail::RadixBitFlagKernel<>
        <<<GridP1, DefaultBlockSize>>>(N, In, Flags, Bit);
    ExclusiveScanInPlace(Flags, N + 1);
    cudaDeviceSynchronize();
    uint32_t ZeroCount = Flags[N];
    detail::RadixBitScatterKernel<>
        <<<Grid, DefaultBlockSize>>>(N, In, Out, Flags, ZeroCount, Bit);
    uint64_t *Tmp = In;
    In = Out;
    Out = Tmp;
  }
  cudaDeviceSynchronize();
  cudaFree(Scratch);
  cudaFree(Flags);
}

} // namespace cuda
} // namespace plastix

#endif // PLASTIX_HAS_CUDA

#endif // PLASTIX_CUDA_PRIMITIVES_HPP
