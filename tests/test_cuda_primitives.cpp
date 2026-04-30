// Smoke tests for the in-house CUDA parallel primitives. Built only when
// PLASTIX_ENABLE_CUDA=ON. The primitives themselves live header-only inside
// `plastix/cuda_primitives.hpp`.

#include <gtest/gtest.h>

#ifdef PLASTIX_HAS_CUDA

#include "plastix/cuda_primitives.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

template <typename T> T *DeviceManaged(size_t N) {
  T *P = nullptr;
  cudaMallocManaged(&P, N * sizeof(T));
  return P;
}

TEST(CudaPrimitives, ExclusiveScanSmall) {
  const size_t N = 8;
  uint32_t *D = DeviceManaged<uint32_t>(N);
  std::vector<uint32_t> Src{3, 1, 4, 1, 5, 9, 2, 6};
  for (size_t I = 0; I < N; ++I)
    D[I] = Src[I];
  plastix::cuda::ExclusiveScanInPlace(D, N);
  cudaDeviceSynchronize();
  std::vector<uint32_t> Expected{0, 3, 4, 8, 9, 14, 23, 25};
  for (size_t I = 0; I < N; ++I)
    EXPECT_EQ(D[I], Expected[I]) << "at " << I;
  cudaFree(D);
}

TEST(CudaPrimitives, ExclusiveScanLarge) {
  const size_t N = 4096;
  uint32_t *D = DeviceManaged<uint32_t>(N);
  for (size_t I = 0; I < N; ++I)
    D[I] = static_cast<uint32_t>(I);
  plastix::cuda::ExclusiveScanInPlace(D, N);
  cudaDeviceSynchronize();
  uint32_t Acc = 0;
  for (size_t I = 0; I < N; ++I) {
    EXPECT_EQ(D[I], Acc) << "at " << I;
    Acc += static_cast<uint32_t>(I);
  }
  cudaFree(D);
}

TEST(CudaPrimitives, AtomicFloatAdd) {
  const size_t N = 1024;
  float *D = DeviceManaged<float>(1);
  D[0] = 0.0f;
  plastix::cuda::AddOnesIntoSlot(D, N);
  cudaDeviceSynchronize();
  EXPECT_NEAR(D[0], static_cast<float>(N), 1e-3f);
  cudaFree(D);
}

TEST(CudaPrimitives, RadixSort64Random) {
  const size_t N = 10'000;
  uint64_t *D = DeviceManaged<uint64_t>(N);
  std::mt19937_64 Rng(0xC0FFEE);
  std::vector<uint64_t> Src(N);
  for (size_t I = 0; I < N; ++I) {
    Src[I] = Rng();
    D[I] = Src[I];
  }
  plastix::cuda::RadixSort64InPlace(D, N);
  cudaDeviceSynchronize();
  std::sort(Src.begin(), Src.end());
  for (size_t I = 0; I < N; ++I)
    EXPECT_EQ(D[I], Src[I]) << "at " << I;
  cudaFree(D);
}

TEST(CudaPrimitives, RadixSort64WithDuplicates) {
  const size_t N = 4096;
  uint64_t *D = DeviceManaged<uint64_t>(N);
  std::mt19937_64 Rng(7);
  std::vector<uint64_t> Src(N);
  for (size_t I = 0; I < N; ++I) {
    Src[I] = Rng() & 0xFFu;
    D[I] = Src[I];
  }
  plastix::cuda::RadixSort64InPlace(D, N);
  cudaDeviceSynchronize();
  std::sort(Src.begin(), Src.end());
  for (size_t I = 0; I < N; ++I)
    EXPECT_EQ(D[I], Src[I]) << "at " << I;
  cudaFree(D);
}

} // namespace

#endif // PLASTIX_HAS_CUDA
