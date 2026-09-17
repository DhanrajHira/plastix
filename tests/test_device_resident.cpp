// Tests for the opt-in device-resident SOA allocator (ADR-006). Built only when
// PLASTIX_ENABLE_CUDA=ON. Exercises the host-staging write -> PushToDevice ->
// kernel mutate via view() -> PullToHost round trip, plus move semantics.

#include <gtest/gtest.h>

#ifdef PLASTIX_HAS_CUDA

#include "plastix/alloc.hpp"
#include "plastix/device_alloc.hpp"

#include <cstdint>
#include <utility>

namespace {

struct WeightTag {};
struct IdTag {};
using WField = plastix::alloc::SOAField<WeightTag, float>;
using IField = plastix::alloc::SOAField<IdTag, uint32_t>;
using DRS = plastix::alloc::DeviceResidentSOA<int, WField, IField>;
using DView = plastix::alloc::SoaView<int, WField, IField>;

// Doubles Weight and adds 100 to Id for each live slot, through the device view.
__global__ void MutateKernel(DView V, size_t N) {
  size_t I = blockIdx.x * blockDim.x + threadIdx.x;
  if (I >= N)
    return;
  V.Get<WeightTag>(I) *= 2.0f;
  V.Get<IdTag>(I) += 100u;
}

TEST(DeviceResidentSOA, RoundTripMutateOnDevice) {
  const size_t N = 256;
  DRS A(N);
  EXPECT_EQ(A.GetCapacity(), N);

  for (size_t I = 0; I < N; ++I) {
    auto Id = A.Allocate();
    ASSERT_NE(Id, static_cast<size_t>(-1));
    A.Get<WeightTag>(Id) = static_cast<float>(I) + 1.0f;
    A.Get<IdTag>(Id) = static_cast<uint32_t>(I);
  }
  EXPECT_EQ(A.Size(), N);

  A.PushToDevice();
  MutateKernel<<<static_cast<unsigned>((N + 63) / 64), 64>>>(A.view(), N);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  A.PullToHost();

  for (size_t I = 0; I < N; ++I) {
    EXPECT_FLOAT_EQ(A.Get<WeightTag>(I), (static_cast<float>(I) + 1.0f) * 2.0f)
        << "at " << I;
    EXPECT_EQ(A.Get<IdTag>(I), static_cast<uint32_t>(I) + 100u) << "at " << I;
  }
}

TEST(DeviceResidentSOA, AllocateAtCapacity) {
  DRS A(4);
  for (int I = 0; I < 4; ++I)
    EXPECT_NE(A.Allocate(), static_cast<size_t>(-1));
  EXPECT_EQ(A.Allocate(), static_cast<size_t>(-1)); // full
  EXPECT_EQ(A.Size(), 4u);
}

TEST(DeviceResidentSOA, MoveTransfersOwnership) {
  DRS A(4);
  auto Id = A.Allocate();
  A.Get<WeightTag>(Id) = 3.0f;

  DRS B(std::move(A));
  EXPECT_EQ(B.Size(), 1u);
  EXPECT_FLOAT_EQ(B.Get<WeightTag>(0), 3.0f);
  // A is moved-from (nulled); its destructor must be a no-op (no double free).
}

} // namespace

#endif // PLASTIX_HAS_CUDA
