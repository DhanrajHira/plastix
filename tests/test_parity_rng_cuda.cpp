// Device half of the RNG contract: the counter-based samplers must produce the
// same result inside a kernel as on the host. Engine-agnostic -- it pins
// host/device agreement of the public entry points rather than any one engine's
// internals.
//
// UniformReal on a power-of-two-width range and SampleKofN (which draws on the
// [0, 1) width-1 range) are bit-exact host vs device: their range mapping is a
// single multiply, so there is no FMA-contraction gap. NormalReal goes through
// transcendentals (log/cos/sqrt) whose device intrinsics differ from libm by a
// few ULP, so it is checked for closeness, not bit-equality.
//
// Gated on PLASTIX_HAS_CUDA, so a CPU-only build compiles this as an empty
// translation unit (mirrors the other *_cuda.cpp tests).

#include <gtest/gtest.h>

#ifdef PLASTIX_HAS_CUDA

#include "plastix/macros.hpp"
#include "plastix/random.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

__host__ __device__ uint32_t Bits(float F) {
  uint32_t B;
  std::memcpy(&B, &F, sizeof(B));
  return B;
}

__global__ void UniformKernel(const uint64_t *Seeds, const uint64_t *Counters,
                              size_t N, float Min, float Max, uint32_t *Out) {
  const size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  Out[I] = Bits(plastix::UniformReal(Seeds[I], Counters[I], Min, Max));
}

__global__ void NormalKernel(const uint64_t *Seeds, const uint64_t *Counters,
                             size_t N, float *Out) {
  const size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  Out[I] = plastix::NormalReal(Seeds[I], Counters[I]);
}

__global__ void KofNKernel(uint64_t Seed, const uint64_t *Counters, size_t N,
                           uint32_t PoolN, uint32_t K, uint32_t *Out) {
  const size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  plastix::SampleKofN(Seed, Counters[I], PoolN, K, Out + I * K);
}

template <class T> struct Managed {
  T *Ptr = nullptr;
  explicit Managed(size_t N) {
    PLASTIX_CUDA_CHECK(cudaMallocManaged(&Ptr, N * sizeof(T)));
  }
  ~Managed() { cudaFree(Ptr); }
  Managed(const Managed &) = delete;
  Managed &operator=(const Managed &) = delete;
};

TEST(ParityRngCuda, UniformMatchesHostOnPowerOfTwoWidth) {
  constexpr size_t N = 8192;
  Managed<uint64_t> Seeds(N), Counters(N);
  Managed<uint32_t> Out(N);
  for (size_t I = 0; I < N; ++I) {
    Seeds.Ptr[I] = 1 + (I % 7);
    Counters.Ptr[I] = I;
  }
  const unsigned Block = 256, Grid = (N + Block - 1) / Block;
  for (const auto [Min, Max] : {std::pair{-1.0f, 1.0f}, std::pair{0.0f, 1.0f}}) {
    UniformKernel<<<Grid, Block>>>(Seeds.Ptr, Counters.Ptr, N, Min, Max,
                                   Out.Ptr);
    PLASTIX_CUDA_CHECK_KERNEL();
    PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t I = 0; I < N; ++I)
      EXPECT_EQ(Out.Ptr[I],
                Bits(plastix::UniformReal(Seeds.Ptr[I], Counters.Ptr[I], Min,
                                          Max)))
          << "seed " << Seeds.Ptr[I] << " counter " << Counters.Ptr[I];
  }
}

TEST(ParityRngCuda, SampleKofNMatchesHost) {
  constexpr size_t N = 2048;
  constexpr uint32_t PoolN = 128, K = 16;
  Managed<uint64_t> Counters(N);
  Managed<uint32_t> Out(N * K);
  for (size_t I = 0; I < N; ++I)
    Counters.Ptr[I] = I;
  const unsigned Block = 256, Grid = (N + Block - 1) / Block;
  KofNKernel<<<Grid, Block>>>(42, Counters.Ptr, N, PoolN, K, Out.Ptr);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  for (size_t I = 0; I < N; ++I) {
    uint32_t Host[K];
    plastix::SampleKofN(42, Counters.Ptr[I], PoolN, K, Host);
    for (uint32_t J = 0; J < K; ++J)
      EXPECT_EQ(Out.Ptr[I * K + J], Host[J]) << "counter " << I << " slot " << J;
  }
}

TEST(ParityRngCuda, NormalIsCloseToHost) {
  constexpr size_t N = 4096;
  Managed<uint64_t> Seeds(N), Counters(N);
  Managed<float> Out(N);
  for (size_t I = 0; I < N; ++I) {
    Seeds.Ptr[I] = 1234;
    Counters.Ptr[I] = I;
  }
  const unsigned Block = 256, Grid = (N + Block - 1) / Block;
  NormalKernel<<<Grid, Block>>>(Seeds.Ptr, Counters.Ptr, N, Out.Ptr);
  PLASTIX_CUDA_CHECK_KERNEL();
  PLASTIX_CUDA_CHECK(cudaDeviceSynchronize());
  for (size_t I = 0; I < N; ++I)
    EXPECT_NEAR(Out.Ptr[I], plastix::NormalReal(Seeds.Ptr[I], Counters.Ptr[I]),
                1e-4f)
        << "counter " << I;
}

} // namespace

#endif // PLASTIX_HAS_CUDA
