// Guards the RNG contract the DEEP R substrate and the plastax conformance
// vectors depend on. Each engine is validated against its own golden (per-engine
// files, no cross-engine matching), plus statistical moments for the derived
// distributions.
//
// Both engine implementations are always compiled (detail:: helpers), so this
// single build checks each uniform golden regardless of which engine the public
// UniformReal dispatches to. NormalReal / SampleKofN are built on the *public*
// UniformReal, so their goldens are per the selected PLASTIX_RNG_ENGINE; the
// bit-exact checks are gated on the golden's "engine" field matching this build.
//
// Uniform comparisons are exact (bit patterns): initial weights are an input to
// every downstream vector, so a one-ULP drift here would silently eat tolerance
// budgets. See notes/parity/00-parity-harness.md.

#include "parity/golden_io.hpp"

#include <plastix/random.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint32_t Bits(float F) {
  uint32_t B;
  std::memcpy(&B, &F, sizeof(B));
  return B;
}

std::string HexBits(float F) {
  char Buf[16];
  std::snprintf(Buf, sizeof(Buf), "%08x", Bits(F));
  return Buf;
}

const char *BuildEngine() {
#if PLASTIX_RNG_ENGINE == PLASTIX_RNG_MINSTD
  return "minstd";
#else
  return "philox";
#endif
}

// Compare an explicit engine sampler against its uniform golden, bit-for-bit.
void CheckUniformGolden(const std::string &File,
                        float (*Sampler)(uint64_t, uint64_t, float, float)) {
  const nlohmann::json Golden = plastix::parity::LoadGolden(File);
  const auto &Ranges = Golden.at("ranges");
  ASSERT_FALSE(Ranges.empty()) << File << " declares no ranges";
  const auto &Samples = Golden.at("samples");
  ASSERT_GT(Samples.size(), 100u) << File << " is suspiciously small";

  size_t Compared = 0;
  for (const auto &Sample : Samples) {
    const auto Seed = Sample.at("seed").get<uint64_t>();
    const auto Counter = Sample.at("counter").get<uint64_t>();
    for (const auto &[Name, Bounds] : Ranges.items()) {
      const float Min = Bounds.at(0).get<float>();
      const float Max = Bounds.at(1).get<float>();
      const std::string Want = Sample.at(Name).get<std::string>();
      const std::string Got = HexBits(Sampler(Seed, Counter, Min, Max));
      EXPECT_EQ(Got, Want)
          << File << ": sampler(" << Seed << ", " << Counter << ", " << Min
          << ", " << Max << ") drifted. Regenerate with emit_rng_golden ONLY if"
          << " the change is intended.";
      ++Compared;
    }
  }
  EXPECT_EQ(Compared, Samples.size() * Ranges.size());
}

TEST(ParityRng, MinstdUniformMatchesGolden) {
  CheckUniformGolden("rng_uniform.json", &plastix::detail::UniformMinstd);
}

TEST(ParityRng, PhiloxUniformMatchesGolden) {
  CheckUniformGolden("rng_uniform_philox.json", &plastix::detail::UniformPhilox);
}

// The range mapping is the part most likely to be ported wrong, so pin the
// property directly rather than trusting the sample grid to cover it.
TEST(ParityRng, SamplesStayWithinTheRequestedRange) {
  for (uint64_t Seed = 0; Seed < 8; ++Seed) {
    for (uint64_t Counter = 0; Counter < 256; ++Counter) {
      const float Signed = plastix::UniformReal(Seed, Counter, -1.0f, 1.0f);
      EXPECT_GE(Signed, -1.0f);
      EXPECT_LT(Signed, 1.0f);
      const float Positive = plastix::UniformReal(Seed, Counter, 2.0f, 5.0f);
      EXPECT_GE(Positive, 2.0f);
      EXPECT_LT(Positive, 5.0f);
    }
  }
}

// The whole point of a counter-based sampler: the same (Seed, Counter) draws
// the same value no matter when or where it is called.
TEST(ParityRng, IsStateless) {
  const float First = plastix::UniformReal(42, 7, -1.0f, 1.0f);
  for (uint64_t Noise = 0; Noise < 100; ++Noise)
    (void)plastix::UniformReal(Noise, Noise, -1.0f, 1.0f);
  EXPECT_EQ(Bits(plastix::UniformReal(42, 7, -1.0f, 1.0f)), Bits(First));
}

// NormalReal is a pure function of the selected engine; pin it bit-for-bit
// against the golden emitted for that engine.
TEST(ParityRng, NormalMatchesGolden) {
  const nlohmann::json Golden = plastix::parity::LoadGolden("rng_normal.json");
  if (Golden.at("engine").get<std::string>() != BuildEngine())
    GTEST_SKIP() << "golden is for engine " << Golden.at("engine")
                 << ", build uses " << BuildEngine();
  const auto &Samples = Golden.at("samples");
  ASSERT_GT(Samples.size(), 100u);
  for (const auto &Sample : Samples) {
    const auto Seed = Sample.at("seed").get<uint64_t>();
    const auto Counter = Sample.at("counter").get<uint64_t>();
    const std::string Want = Sample.at("normal").get<std::string>();
    EXPECT_EQ(HexBits(plastix::NormalReal(Seed, Counter)), Want)
        << "NormalReal(" << Seed << ", " << Counter << ") drifted";
  }
}

// Box-Muller must actually produce a standard normal: mean ~ 0, variance ~ 1.
TEST(ParityRng, NormalHasStandardMoments) {
  constexpr uint64_t N = 200000;
  double Sum = 0.0, SumSq = 0.0;
  for (uint64_t C = 0; C < N; ++C) {
    const double Z = plastix::NormalReal(1234, C);
    Sum += Z;
    SumSq += Z * Z;
  }
  const double Mean = Sum / N;
  const double Var = SumSq / N - Mean * Mean;
  EXPECT_NEAR(Mean, 0.0, 0.02);
  EXPECT_NEAR(Var, 1.0, 0.02);
}

TEST(ParityRng, KofNMatchesGolden) {
  const nlohmann::json Golden = plastix::parity::LoadGolden("rng_kofn.json");
  if (Golden.at("engine").get<std::string>() != BuildEngine())
    GTEST_SKIP() << "golden is for engine " << Golden.at("engine")
                 << ", build uses " << BuildEngine();
  const auto &Samples = Golden.at("samples");
  ASSERT_FALSE(Samples.empty());
  for (const auto &Sample : Samples) {
    const auto Seed = Sample.at("seed").get<uint64_t>();
    const auto Counter = Sample.at("counter").get<uint64_t>();
    const auto N = Sample.at("n").get<uint32_t>();
    const auto K = Sample.at("k").get<uint32_t>();
    std::vector<uint32_t> Out(N ? N : 1);
    plastix::SampleKofN(Seed, Counter, N, K, Out.data());
    const auto Want = Sample.at("out").get<std::vector<uint32_t>>();
    ASSERT_EQ(Want.size(), K);
    for (uint32_t I = 0; I < K; ++I)
      EXPECT_EQ(Out[I], Want[I]) << "SampleKofN(" << Seed << ", " << Counter
                                 << ", " << N << ", " << K << ") drifted at "
                                 << I;
  }
}

// Structural invariants that must hold for every engine: K distinct indices in
// [0, N), strictly ascending.
TEST(ParityRng, KofNIsSortedDistinctInRange) {
  const std::pair<uint32_t, uint32_t> Cases[] = {
      {4, 2}, {8, 3}, {10, 10}, {64, 5}, {256, 16}, {1000, 32}};
  for (const auto &[N, K] : Cases) {
    for (uint64_t Seed = 0; Seed < 4; ++Seed) {
      for (uint64_t Counter = 0; Counter < 64; ++Counter) {
        std::vector<uint32_t> Out(N);
        plastix::SampleKofN(Seed, Counter, N, K, Out.data());
        for (uint32_t I = 0; I < K; ++I)
          EXPECT_LT(Out[I], N);
        for (uint32_t I = 1; I < K; ++I)
          EXPECT_LT(Out[I - 1], Out[I]) << "not strictly ascending / distinct";
      }
    }
  }
}

// Over many draws every index of a small N should get selected: the sampler is
// not biased toward a subset.
TEST(ParityRng, KofNCoversEveryIndex) {
  constexpr uint32_t N = 20, K = 5;
  std::vector<int> Seen(N, 0);
  for (uint64_t Counter = 0; Counter < 20000; ++Counter) {
    uint32_t Out[N];
    plastix::SampleKofN(777, Counter, N, K, Out);
    for (uint32_t I = 0; I < K; ++I)
      ++Seen[Out[I]];
  }
  for (uint32_t I = 0; I < N; ++I)
    EXPECT_GT(Seen[I], 0) << "index " << I << " was never selected";
}

} // namespace
