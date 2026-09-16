#ifndef PLASTIX_DEVICE_RNG_HPP
#define PLASTIX_DEVICE_RNG_HPP

// Host+device RNG for network policies.
//
// Structural policies (AddUnit / AddConn / InitUnit) historically reached for
// std::mt19937, which is host-only and stateful — it cannot run inside a kernel
// and a single shared generator races across threads. To let those policies run
// on-device, use this instead: a stateless, counter-based (SplitMix64) RNG that
// is seeded per *draw site* from (base_seed, key) where `key` is something
// stable and unique per invocation — typically the unit id mixed with the step
// counter. Each thread constructs its own generator, so there is no shared
// state and the stream is deterministic and reproducible across host/device.
//
// Usage in a policy:
//   plastix::DeviceRng R(G.RngSeed, plastix::MixKey(Id, G.Step));
//   bool b = R.Bernoulli(0.5f);
//   int  d = R.UniformInt(1, 20);

#include "plastix/macros.hpp"

#include <cstdint>

namespace plastix {

PLASTIX_HD std::uint64_t SplitMix64(std::uint64_t X) {
  X += 0x9E3779B97F4A7C15ull;
  X = (X ^ (X >> 30)) * 0xBF58476D1CE4E5B9ull;
  X = (X ^ (X >> 27)) * 0x94D049BB133111EBull;
  return X ^ (X >> 31);
}

// Combine two 32-bit-ish ids into one 64-bit key (e.g. unit id + step).
PLASTIX_HD std::uint64_t MixKey(std::uint64_t A, std::uint64_t B) {
  return SplitMix64(A * 0x9E3779B97F4A7C15ull + B + 0x632BE59BD9B4E019ull);
}

// Stateless-per-construction counter RNG. Cheap to copy; no shared state.
class DeviceRng {
public:
  PLASTIX_HD DeviceRng(std::uint64_t Seed, std::uint64_t Key)
      : State_(SplitMix64(Seed ^ MixKey(Key, 0x243F6A8885A308D3ull))) {}

  PLASTIX_HD std::uint32_t NextU32() {
    State_ = SplitMix64(State_);
    return static_cast<std::uint32_t>(State_ >> 32);
  }

  // Uniform float in [0, 1).
  PLASTIX_HD float Uniform() {
    return static_cast<float>(NextU32() >> 8) * (1.0f / 16777216.0f);
  }

  PLASTIX_HD bool Bernoulli(float P) { return Uniform() < P; }

  // Uniform integer in [Lo, Hi] inclusive.
  PLASTIX_HD int UniformInt(int Lo, int Hi) {
    if (Hi <= Lo)
      return Lo;
    std::uint32_t Span = static_cast<std::uint32_t>(Hi - Lo + 1);
    return Lo + static_cast<int>(NextU32() % Span);
  }

private:
  std::uint64_t State_;
};

} // namespace plastix

#endif // PLASTIX_DEVICE_RNG_HPP
