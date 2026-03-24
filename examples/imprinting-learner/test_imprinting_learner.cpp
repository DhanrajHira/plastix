// ===========================================================================
// Tests for the SwiftTD Imprinting Learner
//
// Part 1: SwiftTD policy unit tests (existing)
// Part 2: Nonlinear prediction tests mirroring the Python test suite
//         (run_tests.py). Each test generates data, runs the learner, checks
//         asymptotic loss, and writes CSV for plotting.
// ===========================================================================

#define IMPRINTING_LEARNER_TEST
#include "imprinting_learner.cpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// ===== Part 1: SwiftTD policy unit tests ====================================

namespace {

class SwiftTDTest : public ::testing::Test {
protected:
  static constexpr size_t NInput  = 1;
  static constexpr size_t NOutput = 1;

  ILNet *Net = nullptr;
  ImprintingLearnerAllocator *U = nullptr;
  ImprintingLearnerGlobalState *G = nullptr;

  void SetUp() override {
    Net = new ILNet(NInput, plastix::FullyConnected{NOutput, 0.0f});
    U   = &Net->GetUnitAlloc();
    G   = &Net->GetGlobalState();
    InitFeatureUnit(*U, 0, G->AlphaInit);
    auto &Page = Net->GetConnAlloc().Get<plastix::ConnPageMarker>(0);
    Page.Conn[0].second = 0.0f;
  }

  void TearDown() override { delete Net; }

  float RunStep(float Phi, float Cumulant) {
    U->Get<ValueNodeTag>(0) = Phi;
    float Input[1] = {Phi};
    Net->DoForwardPass(Input);
    float V = Net->GetOutput()[0];
    G->Cumulant = Cumulant;
    G->Delta    = Cumulant + G->GammaDiscount * V - G->V_old;
    Net->DoUpdateUnitState();
    ComputePhase2Globals(*Net, *G, NInput);
    Net->DoUpdateConnectionState();
    G->V_old = V;
    return V;
  }

  float GetWeight() const {
    return Net->GetConnAlloc().Get<plastix::ConnPageMarker>(0).Conn[0].second;
  }
};

TEST_F(SwiftTDTest, ConceptsSatisfied) {
  static_assert(
      plastix::UpdateUnitPolicy<ImprintingLearnerUpdateUnit,
                                ImprintingLearnerAllocator,
                                ImprintingLearnerGlobalState>);
  static_assert(
      plastix::UpdateConnPolicy<ImprintingLearnerUpdateConn,
                                ImprintingLearnerAllocator,
                                plastix::ConnStateAllocator,
                                ImprintingLearnerGlobalState>);
  SUCCEED();
}

TEST_F(SwiftTDTest, Step1InitialTraceUpdate) {
  RunStep(1.0f, 0.0f);
  EXPECT_NEAR(U->Get<ZTag>(0), 1e-7f, 1e-12f);
  EXPECT_NEAR(U->Get<DeltaZTag>(0), 1e-7f, 1e-12f);
  EXPECT_NEAR(U->Get<ZBarTag>(0), 1e-7f, 1e-12f);
  EXPECT_NEAR(GetWeight(), 0.0f, 1e-12f);
}

TEST_F(SwiftTDTest, Step2WeightUpdate) {
  RunStep(1.0f, 0.0f);
  RunStep(1.0f, 1.0f);
  EXPECT_NEAR(GetWeight(), 1e-7f, 1e-10f);
  EXPECT_NEAR(U->Get<BetaTag>(0), std::log(0.5f), 1e-5f);
  EXPECT_GT(U->Get<ZTag>(0), 0.49f);
  EXPECT_LT(U->Get<ZTag>(0), 0.51f);
  EXPECT_NEAR(U->Get<DeltaZTag>(0), 0.5f, 1e-5f);
}

TEST_F(SwiftTDTest, InactiveFeatureNoTraceIncrement) {
  RunStep(1.0f, 0.0f);
  float Z1 = U->Get<ZTag>(0);
  RunStep(0.0f, 0.0f);
  float ExpectedZ = Z1 * (0.1f * 0.95f);
  EXPECT_NEAR(U->Get<ZTag>(0), ExpectedZ, 1e-13f);
  EXPECT_NEAR(U->Get<DeltaZTag>(0), 0.0f, 1e-12f);
}

TEST_F(SwiftTDTest, BetaClampedToLogEta) {
  for (int I = 0; I < 20; ++I)
    RunStep(1.0f, 0.0f);
  EXPECT_LE(U->Get<BetaTag>(0), std::log(G->Eta) + 1e-5f);
}

// ===== Part 2: Data generation (matching Python run_tests.py) ===============

std::vector<std::vector<float>>
GenerateRandomBinary(size_t NSteps, std::mt19937 &Rng) {
  std::uniform_int_distribution<int> Dist(0, 1);
  std::vector<std::vector<float>> Data(NSteps, std::vector<float>(2));
  for (size_t T = 0; T < NSteps; ++T) {
    Data[T][0] = static_cast<float>(Dist(Rng));
    Data[T][1] = static_cast<float>(Dist(Rng));
  }
  return Data;
}

std::vector<std::vector<float>>
GeneratePulsePattern(size_t NSteps, size_t Wait, size_t PulseLength = 4) {
  // Phases: [1,0]*PL, [0,0]*Wait, [0,1]*PL, [0,0]*Wait, [1,1]*PL, [0,0]*Wait
  struct Phase { size_t Dur; float A, B; };
  std::vector<Phase> Phases = {
      {PulseLength, 1, 0}, {Wait, 0, 0},
      {PulseLength, 0, 1}, {Wait, 0, 0},
      {PulseLength, 1, 1}, {Wait, 0, 0},
  };
  std::vector<std::vector<float>> Cycle;
  for (auto &P : Phases) {
    for (size_t I = 0; I < P.Dur; ++I)
      Cycle.push_back({P.A, P.B});
  }
  std::vector<std::vector<float>> Data(NSteps, std::vector<float>(2));
  for (size_t T = 0; T < NSteps; ++T)
    Data[T] = Cycle[T % Cycle.size()];
  return Data;
}

// ===== Reward functions =====================================================

std::vector<float>
ComputeRewardsLinearLag(const std::vector<std::vector<float>> &Inputs,
                        size_t Lag) {
  size_t N = Inputs.size();
  std::vector<float> Rewards(N, 0.0f);
  for (size_t T = Lag; T < N; ++T)
    Rewards[T] = Inputs[T - Lag][0] - Inputs[T - Lag][1];
  return Rewards;
}

std::vector<float>
ComputeRewardsXorLag(const std::vector<std::vector<float>> &Inputs,
                     size_t Lag) {
  size_t N = Inputs.size();
  std::vector<float> Rewards(N, 0.0f);
  for (size_t T = Lag; T < N; ++T) {
    int a = static_cast<int>(Inputs[T - Lag][0]);
    int b = static_cast<int>(Inputs[T - Lag][1]);
    Rewards[T] = static_cast<float>(a ^ b);
  }
  return Rewards;
}

// ===== Return computation ===================================================

std::vector<float> ComputeReturns(const std::vector<float> &Rewards,
                                  float Gamma) {
  size_t N = Rewards.size();
  std::vector<float> Returns(N, 0.0f);
  for (size_t T = N - 1; T > 0; --T)
    Returns[T - 1] = Rewards[T] + Gamma * Returns[T];
  return Returns;
}

// ===== Test runner ==========================================================

struct TestResult {
  std::vector<float> Predictions;
  std::vector<float> Errors;
  float AsymptoticLoss;
  size_t FinalFeatureCount;
};

TestResult RunLearnerTest(ImprintingLearner &Learner,
                          const std::vector<std::vector<float>> &Inputs,
                          const std::vector<float> &Rewards,
                          const std::vector<float> &Returns) {
  size_t N = Inputs.size();
  TestResult R;
  R.Predictions.resize(N);
  R.Errors.resize(N);

  for (size_t T = 0; T < N; ++T) {
    auto [Pred, Delta] = Learner.Step(Inputs[T], Rewards[T]);
    R.Predictions[T] = Pred;
    R.Errors[T] = (Pred - Returns[T]) * (Pred - Returns[T]);
  }

  size_t Tail = static_cast<size_t>(0.9 * static_cast<double>(N));
  float Sum = 0.0f;
  for (size_t T = Tail; T < N; ++T)
    Sum += R.Errors[T];
  R.AsymptoticLoss = Sum / static_cast<float>(N - Tail);
  R.FinalFeatureCount = Learner.NumTotalFeatures();
  return R;
}

void WriteCSV(const std::string &Path,
              const std::vector<float> &Predictions,
              const std::vector<float> &Returns,
              const std::vector<float> &Errors) {
  std::ofstream F(Path);
  F << "step,prediction,return,error\n";
  for (size_t T = 0; T < Predictions.size(); ++T) {
    F << T << "," << Predictions[T] << "," << Returns[T] << ","
      << Errors[T] << "\n";
  }
}

// ===== Test configurations ==================================================

struct TestConfig {
  const char *Name;
  float Gamma;
  bool UsePulse;
  size_t Wait;
  bool UseXor;
  size_t RewardLag;
  bool GenPattern;
  bool GenMemory;
  float TargetLoss;
};

// ===== Nonlinear prediction tests ===========================================

class NonlinearPredictionTest
    : public ::testing::TestWithParam<TestConfig> {};

TEST_P(NonlinearPredictionTest, AsymptoticLoss) {
  auto Cfg = GetParam();
  constexpr size_t NSteps = 20000;

  std::mt19937 Rng(12345);

  auto Inputs = Cfg.UsePulse
                    ? GeneratePulsePattern(NSteps, Cfg.Wait)
                    : GenerateRandomBinary(NSteps, Rng);
  auto Rewards = Cfg.UseXor
                     ? ComputeRewardsXorLag(Inputs, Cfg.RewardLag)
                     : ComputeRewardsLinearLag(Inputs, Cfg.RewardLag);
  auto Returns = ComputeReturns(Rewards, Cfg.Gamma);

  ImprintingLearner::Config LCfg;
  LCfg.NumObs         = 2;
  LCfg.GenerationLimit = 5;
  LCfg.GenPattern      = Cfg.GenPattern;
  LCfg.GenMemory       = Cfg.GenMemory;
  LCfg.Gamma           = Cfg.Gamma;
  LCfg.Seed            = 42;

  ImprintingLearner Learner(LCfg);
  auto Result = RunLearnerTest(Learner, Inputs, Rewards, Returns);

  // Write CSV for plotting
  std::string CsvName = std::string("test_") + Cfg.Name + ".csv";
  // Replace spaces with underscores
  for (auto &Ch : CsvName)
    if (Ch == ' ' || Ch == ',') Ch = '_';
  WriteCSV(CsvName, Result.Predictions, Returns, Result.Errors);

  std::cout << "  " << Cfg.Name
            << ": asymptotic_loss=" << Result.AsymptoticLoss
            << " (target=" << Cfg.TargetLoss << ")"
            << " features=" << Result.FinalFeatureCount << "\n";

  // Check asymptotic loss within tolerance
  float AbsDiff = std::abs(Result.AsymptoticLoss - Cfg.TargetLoss);
  bool AbsOk = AbsDiff <= 1e-3f;
  bool RelOk = Cfg.TargetLoss != 0.0f &&
               AbsDiff / std::abs(Cfg.TargetLoss) <= 0.2f;
  EXPECT_TRUE(AbsOk || RelOk)
      << "Asymptotic loss " << Result.AsymptoticLoss
      << " not close enough to target " << Cfg.TargetLoss;
}

// Test 1: Linear, gamma=0, lag=1 — obs features alone suffice
// Test 3: Pattern+memory, gamma=0, XOR lag=2 — pattern features detect XOR,
//          memory features resolve temporal lag at phase boundaries
// Test 5: Memory, gamma=0, lag=4 — chained delay-1 memory features
static TestConfig kTestConfigs[] = {
    {"one_step_linear",  0.0f, false, 0, false, 1, false, false, 0.0f},
    {"one_step_pattern", 0.0f, true,  4, true,  2, true,  true,  0.0f},
    {"one_step_memory",  0.0f, false, 0, false, 4, false, true,  0.0f},
};

INSTANTIATE_TEST_SUITE_P(
    ImprintingLearner, NonlinearPredictionTest,
    ::testing::ValuesIn(kTestConfigs),
    [](const ::testing::TestParamInfo<TestConfig> &Info) {
      return std::string(Info.param.Name);
    });

} // namespace
