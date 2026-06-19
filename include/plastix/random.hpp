#ifndef PLASTIX_RANDOM_HPP
#define PLASTIX_RANDOM_HPP

#include "plastix/macros.hpp"

// CCCL's thrust::random is used only for CUDA builds, where it gives us an RNG
// that runs identically in host and device code. For CPU-only builds we drop
// the CCCL dependency entirely and use the self-contained replica below, which
// is bit-for-bit identical to the thrust path (same minstd engine, same
// [0,1) mapping, same lerp) — verified across the full (seed, counter) space.
#ifdef PLASTIX_HAS_CUDA
#include <thrust/random.h>
#endif

#include <cstdint>

namespace plastix {

// Counter-based wrappers around a minstd LCG. The framework's initialisers are
// stateless on purpose: weight/bias initialisation has to behave the same
// whether it runs on host or inside a kernel, and a connection at id `c` must
// always draw the same sample regardless of scheduling. We provide the
// pure-function surface here on top of a per-call engine seeded from a hash of
// (Seed, Counter).

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
// --- CCCL-free uniform sampler (CPU builds) -------------------------------
//
// Faithful reimplementation of thrust::default_random_engine (minstd_rand,
// a=48271, c=0, m=2^31-1) plus thrust::uniform_real_distribution<float>.
// Kept byte-identical so CPU and CUDA builds produce the same numbers.
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
  uint32_t Draw =
      static_cast<uint32_t>((static_cast<uint64_t>(kMinstdA) * State) % kMinstdM);
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
  return detail::UniformFromSeed(detail::MixSeed(Seed, Counter), 0.0f, 1.0f) < P;
#endif
}

} // namespace plastix

#endif // PLASTIX_RANDOM_HPP
