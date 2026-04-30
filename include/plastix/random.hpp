#ifndef PLASTIX_RANDOM_HPP
#define PLASTIX_RANDOM_HPP

#include "plastix/macros.hpp"

#include <cstdint>

namespace plastix {

// Counter-based Philox4x32-10 RNG, host- and device-callable.
//
// Stateless: takes (seed, counter) and returns a deterministic 4x32-bit
// vector. The counter is typically the unit/connection id; the seed is
// per-network. We hand-roll this to avoid a curand dependency and to
// guarantee bit-identical sequences between host and device builds.
//
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3",
// SC'11. The constants are the standard Philox-4x32 multipliers and
// Weyl-sequence constants.

struct Philox4x32 {
  uint32_t X0, X1, X2, X3;
};

namespace detail {

// 32x32 -> high32 helper. nvcc has __umulhi; on host fall back to a 64-bit
// widen. The two paths must be bit-equivalent.
PLASTIX_HD uint32_t MulHi32(uint32_t A, uint32_t B) {
#if defined(__CUDA_ARCH__)
  return __umulhi(A, B);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(A) *
                                static_cast<uint64_t>(B)) >>
                               32);
#endif
}

PLASTIX_HD void Round(uint32_t &X0, uint32_t &X1, uint32_t &X2, uint32_t &X3,
                      uint32_t K0, uint32_t K1) {
  // Standard Philox-4x32 round: paired multiply-and-swap.
  constexpr uint32_t M0 = 0xD2511F53u;
  constexpr uint32_t M1 = 0xCD9E8D57u;
  uint32_t Lo0 = M0 * X0;
  uint32_t Hi0 = MulHi32(M0, X0);
  uint32_t Lo1 = M1 * X2;
  uint32_t Hi1 = MulHi32(M1, X2);
  uint32_t NX0 = Hi1 ^ X1 ^ K0;
  uint32_t NX1 = Lo1;
  uint32_t NX2 = Hi0 ^ X3 ^ K1;
  uint32_t NX3 = Lo0;
  X0 = NX0;
  X1 = NX1;
  X2 = NX2;
  X3 = NX3;
}

} // namespace detail

// 10-round Philox-4x32. The Weyl constants W0/W1 advance the key each round.
PLASTIX_HD Philox4x32 Philox(uint64_t Seed, uint64_t Counter) {
  uint32_t X0 = static_cast<uint32_t>(Counter);
  uint32_t X1 = static_cast<uint32_t>(Counter >> 32);
  uint32_t X2 = static_cast<uint32_t>(Seed);
  uint32_t X3 = static_cast<uint32_t>(Seed >> 32);
  uint32_t K0 = static_cast<uint32_t>(Seed);
  uint32_t K1 = static_cast<uint32_t>(Seed >> 32);
  constexpr uint32_t W0 = 0x9E3779B9u;
  constexpr uint32_t W1 = 0xBB67AE85u;
  for (int R = 0; R < 10; ++R) {
    detail::Round(X0, X1, X2, X3, K0, K1);
    K0 += W0;
    K1 += W1;
  }
  return {X0, X1, X2, X3};
}

// Convert a uint32 sample into a float in [0, 1) with 24 bits of precision
// (the float mantissa width). The mask zeroes the low byte before scaling.
PLASTIX_HD float UnitFloat(uint32_t Bits) {
  return static_cast<float>(Bits >> 8) * (1.0f / 16777216.0f);
}

// Distribution helpers — counter-based, no shared state.
PLASTIX_HD float UniformReal(uint64_t Seed, uint64_t Counter, float Min = 0.0f,
                             float Max = 1.0f) {
  Philox4x32 R = Philox(Seed, Counter);
  return Min + (Max - Min) * UnitFloat(R.X0);
}

PLASTIX_HD bool Bernoulli(uint64_t Seed, uint64_t Counter, float P) {
  Philox4x32 R = Philox(Seed, Counter);
  return UnitFloat(R.X0) < P;
}

} // namespace plastix

#endif // PLASTIX_RANDOM_HPP
