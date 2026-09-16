#ifndef PLASTIX_RANDOM_HPP
#define PLASTIX_RANDOM_HPP

#include "plastix/macros.hpp"

// Counter-based RNG substrate. Stateless pure functions of (Seed, Counter):
// id `c` always draws the same sample on host or device, regardless of
// scheduling. Two engines are provided and selected at compile time:
//
//   PLASTIX_RNG_PHILOX  (default) -- self-contained Philox-4x32-10, pure
//                        integer arithmetic, needs no thrust/CCCL on any
//                        backend, and has stronger statistics than minstd.
//   PLASTIX_RNG_MINSTD  -- thrust::default_random_engine parity path (a minstd
//                        LCG); delegates to thrust on device, and uses a
//                        byte-identical CCCL-free replica on host builds.
//
// Override by defining PLASTIX_RNG_ENGINE before including this header. Both
// engine implementations are always compiled as detail:: helpers so a single
// build can validate each against its own golden; only the public entry points
// dispatch on the macro.
#define PLASTIX_RNG_PHILOX 1
#define PLASTIX_RNG_MINSTD 2
#ifndef PLASTIX_RNG_ENGINE
#define PLASTIX_RNG_ENGINE PLASTIX_RNG_PHILOX
#endif

// thrust is CUDA-only and only the minstd engine can reach for it; a CPU-only
// build never includes it (Philox needs no CCCL, and the host minstd path uses
// the replica below).
#ifdef PLASTIX_HAS_CUDA
#include <thrust/random.h>
#endif

#include <cmath>
#include <cstdint>

namespace plastix {

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

// -------------------------------------------------------------------------
// Philox-4x32-10 (Salmon et al., SC'11). Self-contained, host/device, no CCCL.
// -------------------------------------------------------------------------

// 32x32 -> high 32 bits. nvcc has __umulhi; on host widen to 64 bits. The two
// paths are bit-equivalent.
PLASTIX_HD uint32_t MulHi32(uint32_t A, uint32_t B) {
#if defined(__CUDA_ARCH__)
  return __umulhi(A, B);
#else
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(A) * static_cast<uint64_t>(B)) >> 32);
#endif
}

// One standard Philox-4x32 round: paired multiply-and-swap.
PLASTIX_HD void PhiloxRound(uint32_t &X0, uint32_t &X1, uint32_t &X2,
                            uint32_t &X3, uint32_t K0, uint32_t K1) {
  constexpr uint32_t M0 = 0xD2511F53u;
  constexpr uint32_t M1 = 0xCD9E8D57u;
  uint32_t Lo0 = M0 * X0;
  uint32_t Hi0 = MulHi32(M0, X0);
  uint32_t Lo1 = M1 * X2;
  uint32_t Hi1 = MulHi32(M1, X2);
  X0 = Hi1 ^ X1 ^ K0;
  X1 = Lo1;
  X2 = Hi0 ^ X3 ^ K1;
  X3 = Lo0;
}

// 10-round Philox-4x32; the Weyl constants advance the key each round. Returns
// the first output word, which is all the scalar distributions below consume.
PLASTIX_HD uint32_t Philox32(uint64_t Seed, uint64_t Counter) {
  uint32_t X0 = static_cast<uint32_t>(Counter);
  uint32_t X1 = static_cast<uint32_t>(Counter >> 32);
  uint32_t X2 = static_cast<uint32_t>(Seed);
  uint32_t X3 = static_cast<uint32_t>(Seed >> 32);
  uint32_t K0 = static_cast<uint32_t>(Seed);
  uint32_t K1 = static_cast<uint32_t>(Seed >> 32);
  constexpr uint32_t W0 = 0x9E3779B9u;
  constexpr uint32_t W1 = 0xBB67AE85u;
  for (int R = 0; R < 10; ++R) {
    PhiloxRound(X0, X1, X2, X3, K0, K1);
    K0 += W0;
    K1 += W1;
  }
  return X0;
}

// uint32 -> [0, 1) with 24 bits of mantissa precision.
PLASTIX_HD float UnitFloat(uint32_t Bits) {
  return static_cast<float>(Bits >> 8) * (1.0f / 16777216.0f);
}

PLASTIX_HD float UniformPhilox(uint64_t Seed, uint64_t Counter, float Min,
                               float Max) {
  return Min + (Max - Min) * UnitFloat(Philox32(Seed, Counter));
}

// -------------------------------------------------------------------------
// minstd (thrust::default_random_engine parity path).
// -------------------------------------------------------------------------

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

PLASTIX_HD float UniformMinstd(uint64_t Seed, uint64_t Counter, float Min,
                               float Max) {
#ifdef PLASTIX_HAS_CUDA
  thrust::default_random_engine Eng(MixSeed(Seed, Counter));
  thrust::uniform_real_distribution<float> Dist(Min, Max);
  return Dist(Eng);
#else
  return UniformFromSeed(MixSeed(Seed, Counter), Min, Max);
#endif
}

// Injective (Counter, Sub) -> uint64 for Counter < 2^32: Counter occupies the
// high 32 bits, Sub the low 32. Philox decorrelates adjacent sub-counters, so
// this is a safe way to fan a single call site out into independent draws.
PLASTIX_HD uint64_t SubCounter(uint64_t Counter, uint32_t Sub) {
  return (Counter << 32) | static_cast<uint64_t>(Sub);
}

} // namespace detail

// Dispatch on the selected engine.
PLASTIX_HD float UniformReal(uint64_t Seed, uint64_t Counter, float Min = 0.0f,
                             float Max = 1.0f) {
#if PLASTIX_RNG_ENGINE == PLASTIX_RNG_MINSTD
  return detail::UniformMinstd(Seed, Counter, Min, Max);
#else
  return detail::UniformPhilox(Seed, Counter, Min, Max);
#endif
}

PLASTIX_HD bool Bernoulli(uint64_t Seed, uint64_t Counter, float P) {
  return UniformReal(Seed, Counter, 0.0f, 1.0f) < P;
}

// Standard-normal draw via deterministic Box-Muller on two uniforms. The two
// uniforms are drawn at sub-counters 2*Counter and 2*Counter+1, so NormalReal
// strides the uniform counter space by 2: a caller must keep Counter < 2^63 and
// must not also draw UniformReal at 2*Counter / 2*Counter+1 in the same phase.
PLASTIX_HD float NormalReal(uint64_t Seed, uint64_t Counter) {
  const float U1 = UniformReal(Seed, 2ull * Counter);
  const float U2 = UniformReal(Seed, 2ull * Counter + 1ull);
  // 1 - U1 lies in (0, 1], so the log is always finite (U1 == 1 is impossible).
  const float R = std::sqrt(-2.0f * std::log(1.0f - U1));
  return R * std::cos(6.28318530717958648f * U2);
}

// K distinct indices in [0, N), uniform without replacement, sorted ascending.
// Floyd's algorithm driven by UniformReal, using the Out array as the ordered
// working set (no std containers). Draw i consumes UniformReal at the packed
// sub-counter (Counter << 32) | i, so distinct Counters give disjoint draw
// streams for any K as long as Counter < 2^32. Out must hold >= K entries.
PLASTIX_HD void SampleKofN(uint64_t Seed, uint64_t Counter, uint32_t N,
                           uint32_t K, uint32_t *Out) {
  if (K > N)
    K = N;
  uint32_t Count = 0;
  uint32_t Draw = 0;
  for (uint32_t J = N - K; J < N; ++J) {
    const uint32_t Range = J + 1u;
    const float U = UniformReal(Seed, detail::SubCounter(Counter, Draw));
    ++Draw;
    uint32_t T = static_cast<uint32_t>(U * static_cast<float>(Range));
    if (T >= Range)
      T = Range - 1u; // guard the U -> 1 rounding edge
    uint32_t Pos = 0;
    while (Pos < Count && Out[Pos] < T)
      ++Pos;
    if (Pos < Count && Out[Pos] == T) {
      // T already chosen; J is strictly greater than every current entry.
      Out[Count] = J;
    } else {
      for (uint32_t M = Count; M > Pos; --M)
        Out[M] = Out[M - 1u];
      Out[Pos] = T;
    }
    ++Count;
  }
}

} // namespace plastix

#endif // PLASTIX_RANDOM_HPP
