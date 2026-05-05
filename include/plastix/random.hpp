#ifndef PLASTIX_RANDOM_HPP
#define PLASTIX_RANDOM_HPP

#include "plastix/macros.hpp"

#include <thrust/random.h>

#include <cstdint>

namespace plastix {

// Counter-based wrappers around CCCL's thrust::random. The framework's
// initialisers are stateless on purpose: weight/bias initialisation has to
// behave the same whether it runs on host or inside a kernel, and a
// connection at id `c` must always draw the same sample regardless of
// scheduling. Thrust's engines are themselves stateful, so we provide the
// pure-function surface here on top of a per-call engine seeded from a
// hash of (Seed, Counter).

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

} // namespace detail

PLASTIX_HD float UniformReal(uint64_t Seed, uint64_t Counter, float Min = 0.0f,
                             float Max = 1.0f) {
  thrust::default_random_engine Eng(detail::MixSeed(Seed, Counter));
  thrust::uniform_real_distribution<float> Dist(Min, Max);
  return Dist(Eng);
}

PLASTIX_HD bool Bernoulli(uint64_t Seed, uint64_t Counter, float P) {
  thrust::default_random_engine Eng(detail::MixSeed(Seed, Counter));
  thrust::uniform_real_distribution<float> Dist(0.0f, 1.0f);
  return Dist(Eng) < P;
}

} // namespace plastix

#endif // PLASTIX_RANDOM_HPP
