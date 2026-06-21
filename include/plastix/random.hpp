#ifndef PLASTIX_RANDOM_HPP
#define PLASTIX_RANDOM_HPP

#include "plastix/macros.hpp"

#include <cstdint>

#ifdef PLASTIX_HAS_CUDA
#include <thrust/random.h>
#else
#include <cmath>
#endif

namespace plastix {

// Counter-based RNG wrappers. The framework's initialisers are stateless on
// purpose: weight/bias initialisation has to behave the same whether it runs
// on host or inside a kernel, and a connection at id `c` must always draw the
// same sample regardless of scheduling. Each call seeds a fresh engine from a
// hash of (Seed, Counter) and draws a single value.
//
// In CUDA builds this delegates to CCCL's thrust::random. Host-only builds
// must not pull in thrust/CUDA headers at all, so they use a self-contained
// sampler (below) that reproduces the thrust path bit-for-bit — host and
// device builds therefore initialise identically.

namespace detail {

// SplitMix64 finalizer over a Weyl-mixed (Seed, Counter) pair. Cheap, fully
// host/device portable, and gives every distinct (Seed, Counter) a well-
// decorrelated initial state for the downstream LCG.
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

// Host counterpart to `thrust::default_random_engine` (== minstd_rand) seeded
// from MixSeed, plus `thrust::uniform_real_distribution<float>`, collapsed to a
// single pure call. Replicated exactly so a host build matches a CUDA build's
// weights without depending on thrust:
//   * seed:   m_x = (s mod m) or 1 if that is 0   (minstd, c == 0 guard)
//   * draw:   m_x = (a * m_x) mod m
//   * map:    (m_x - min) / (1 + (max - min)) in [0, 1), then std::lerp
// where minstd has a = 48271, m = 2^31 - 1, min = 1, max = m - 1.
PLASTIX_HD float SampleUniform(uint64_t Seed, uint64_t Counter, float Min,
                               float Max) {
  constexpr uint32_t M = 2147483647u; // 2^31 - 1
  constexpr uint32_t A = 48271u;
  uint32_t X = MixSeed(Seed, Counter) % M;
  if (X == 0u)
    X = 1u;
  X = static_cast<uint32_t>((static_cast<uint64_t>(A) * X) % M);
  // Denominator is 1 + (max - min) = 1 + ((M - 1) - 1) = M - 1.
  float Result = static_cast<float>(X - 1u) / 2147483646.0f;
  return std::lerp(Min, Max, Result);
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
  return detail::SampleUniform(Seed, Counter, Min, Max);
#endif
}

PLASTIX_HD bool Bernoulli(uint64_t Seed, uint64_t Counter, float P) {
#ifdef PLASTIX_HAS_CUDA
  thrust::default_random_engine Eng(detail::MixSeed(Seed, Counter));
  thrust::uniform_real_distribution<float> Dist(0.0f, 1.0f);
  return Dist(Eng) < P;
#else
  return detail::SampleUniform(Seed, Counter, 0.0f, 1.0f) < P;
#endif
}

} // namespace plastix

#endif // PLASTIX_RANDOM_HPP
