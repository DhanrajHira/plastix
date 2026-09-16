#ifndef PLASTIX_RANDOM_HPP
#define PLASTIX_RANDOM_HPP

#include "plastix/macros.hpp"

// CUDA builds use thrust::random; CPU builds use the self-contained replica
// below, bit-for-bit identical to thrust (same minstd, [0,1) map, lerp) across
// all (seed,counter).
#ifdef PLASTIX_HAS_CUDA
#include <thrust/random.h>
#endif

#include <cstdint>

namespace plastix {

// Counter-based wrappers around a minstd LCG: stateless pure functions so id
// `c` always draws the same sample on host or device, seeded from a hash of
// (Seed, Counter).

namespace detail {

// SplitMix64 finalizer over a Weyl-mixed (Seed, Counter) pair: cheap,
// host/device portable, decorrelates every distinct (Seed, Counter) for the
// downstream LCG.
PLASTIX_HD uint32_t MixSeed(uint64_t Seed, uint64_t Counter) {
  uint64_t X = Seed + 0x9E3779B97F4A7C15ull * (Counter + 1ull);
  X ^= X >> 33;
  X *= 0xFF51AFD7ED558CCDull;
  X ^= X >> 33;
  X *= 0xC4CEB9FE1A85EC53ull;
  X ^= X >> 33;
  return static_cast<uint32_t>(X);
}

#ifndef PLASTIX_HAS_CUDA
// CCCL-free uniform sampler (CPU builds): reimplements thrust minstd_rand
// (a=48271,c=0,m=2^31-1) + uniform_real<float>, byte-identical to the CUDA
// path.
constexpr uint32_t kMinstdA = 48271u;
constexpr uint32_t kMinstdM = 2147483647u; // 2^31 - 1
constexpr uint32_t kMinstdMin = 1u;        // engine min (c == 0)
constexpr uint32_t kMinstdMax = kMinstdM - 1u;

// Standard C++20 lerp algorithm (matches cuda::std::lerp used by thrust).
PLASTIX_HD float Lerp(float A, float B, float T) {
  if ((A <= 0.0f && B >= 0.0f) || (A >= 0.0f && B <= 0.0f)) {
    return T * B + (1.0f - T) * A;
  }
  if (T == 1.0f) {
    return B;
  }
  const float X = A + T * (B - A);
  if ((T > 1.0f) == (B > A)) {
    return B < X ? X : B;
  }
  return X < B ? X : B;
}

// One minstd draw seeded as thrust seeds it, mapped to [Min, Max).
PLASTIX_HD float UniformFromSeed(uint32_t S, float Min, float Max) {
  uint32_t State = (S % kMinstdM == 0u) ? 1u : (S % kMinstdM);
  uint32_t Draw = static_cast<uint32_t>(
      (static_cast<uint64_t>(kMinstdA) * State) % kMinstdM);
  float R = static_cast<float>(Draw - kMinstdMin);
  R /= (1.0f + static_cast<float>(kMinstdMax - kMinstdMin));
  return Lerp(Min, Max, R);
}
#endif // !PLASTIX_HAS_CUDA

} // namespace detail

PLASTIX_HD float UniformReal(uint64_t Seed, uint64_t Counter, float Min = 0.0f,
                             float Max = 1.0f) {
#ifdef PLASTIX_HAS_CUDA
  thrust::default_random_engine Eng(detail::MixSeed(Seed, Counter));
  thrust::uniform_real_distribution<float> Dist(Min, Max);
  return Dist(Eng);
#else
  return detail::UniformFromSeed(detail::MixSeed(Seed, Counter), Min, Max);
#endif
}

PLASTIX_HD bool Bernoulli(uint64_t Seed, uint64_t Counter, float P) {
#ifdef PLASTIX_HAS_CUDA
  thrust::default_random_engine Eng(detail::MixSeed(Seed, Counter));
  thrust::uniform_real_distribution<float> Dist(0.0f, 1.0f);
  return Dist(Eng) < P;
#else
  return detail::UniformFromSeed(detail::MixSeed(Seed, Counter), 0.0f, 1.0f) <
         P;
#endif
}

} // namespace plastix

#endif // PLASTIX_RANDOM_HPP
