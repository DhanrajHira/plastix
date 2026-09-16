// Time-to-quality microbench.
//
// Wall-clock vs error curves on four synthetic tasks chosen to mirror
// the families in traditional/{01, 02, 04, 05}:
//
//   static            : fit a fixed batch (offline regression)
//   idempotent        : map x -> x through a hidden bottleneck
//   continuous_small  : online stream with low input dim and concept drift
//   continuous_large  : online stream with delayed-feedback target (MG-like)
//
// Each (task, framework) pair trains for a wall-clock budget and emits
// (elapsed_ms, mse_train, mse_holdout) checkpoints every ~100 ms. The
// frameworks compared are Plastix (always) and a torch::nn::Sequential
// MLP (when PLASTIX_HAVE_TORCH is defined).
//
// Validation: tiny instances of every task with closed-form expected
// values for the target function. Asserted before any training begins.
//
// Output: build/benchmarks/_outputs/time_to_quality.csv (long format,
// one row per checkpoint per (task, framework)).

#include <benchmark/benchmark.h>

#include "_bench_common.hpp"
#include "plastix/plastix.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

#ifdef PLASTIX_HAVE_TORCH
#include <torch/torch.h>
#endif

using bench_common::AppendRow;
using bench_common::CsvRow;
using bench_common::EdgeExists;
using bench_common::EdgeWeight;
using bench_common::OutputFile;
using bench_common::RandomVector;

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;

namespace {

// ---------------------------------------------------------------------------
// Target functions. Each takes a fixed-dim input vector and writes a
// fixed-dim target vector. Pure / deterministic / closed-form so the
// validation step can hand-check small instances.
// ---------------------------------------------------------------------------

// Bench 01 analogue: fixed regression target. y = sum_i (w_i * x_i) with
// hand-picked weights, no noise; the network has to discover a smooth
// linear-plus-bias mapping.
void TargetStatic(std::span<const float> X, std::span<float> Y) {
  // Y is dimension 1.
  float W[8] = {0.5f, -0.3f, 0.2f, 0.1f, -0.4f, 0.25f, 0.15f, -0.05f};
  float Acc = 0.0f;
  for (size_t I = 0; I < X.size() && I < 8; ++I)
    Acc += W[I] * X[I];
  Y[0] = Acc;
}

// Bench 02 analogue: idempotent (identity) target. y = x. Tests whether
// the network can pass information through a narrow hidden bottleneck.
void TargetIdempotent(std::span<const float> X, std::span<float> Y) {
  for (size_t I = 0; I < Y.size(); ++I)
    Y[I] = (I < X.size()) ? X[I] : 0.0f;
}

// Bench 04 analogue: a small streaming target with a slow time drift.
// y = sin(2 * pi * (x0 + phase)). phase increments every batch so the
// network must adapt online.
struct ContinuousSmallState {
  float Phase = 0.0f;
  void Step() { Phase += 0.005f; }
};

void TargetContinuousSmall(std::span<const float> X, std::span<float> Y,
                           const ContinuousSmallState &S) {
  float V = X.empty() ? 0.0f : X[0];
  Y[0] = std::sin(2.0f * 3.14159265f * (V + S.Phase));
}

// Bench 05 analogue: Mackey-Glass-ish delayed feedback. y_t = 0.5 * y_{t-d}
// + nonlin(x_t). Uses an internal ring buffer so the harness drives the
// recurrence outside the network.
struct ContinuousLargeState {
  static constexpr size_t Delay = 16;
  std::vector<float> Ring = std::vector<float>(Delay, 0.0f);
  size_t Cursor = 0;
  float Step(float XSum) {
    float Y = 0.5f * Ring[Cursor] + std::tanh(0.3f * XSum);
    Ring[Cursor] = Y;
    Cursor = (Cursor + 1) % Delay;
    return Y;
  }
};

// ---------------------------------------------------------------------------
// Plastix traits: linear MLP with MSE + SGD, no structural mutation
// (we are comparing pure optimization wall-clock, not topology growth).
// ---------------------------------------------------------------------------

struct GradPreActTag {};

struct LossOnlyGlobals {
  float LearningRate = 0.01f;
};

struct LinearForward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t Cid, auto &) {
    return plastix::GetWeight(C, Cid) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetActivation(U, Id) = Acc;
  }
};

struct LinearBackward {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t ToId, auto &C,
                              size_t Cid, auto &) {
    return plastix::GetWeight(C, Cid) *
           plastix::GetField<GradPreActTag>(U, ToId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetField<GradPreActTag>(U, Id) = Acc;
  }
};

struct SGDUpdateConn {
  PLASTIX_HD static void
  UpdateIncomingConnection(auto &U, size_t Dst, size_t Src, auto &C,
                           size_t Cid, LossOnlyGlobals &G) {
    float Grad = plastix::GetField<GradPreActTag>(U, Dst);
    float In = plastix::GetActivation(U, Src);
    plastix::GetWeight(C, Cid) -= G.LearningRate * Grad * In;
  }
  PLASTIX_HD static void UpdateOutgoingConnection(auto &, size_t, size_t,
                                                  auto &, size_t,
                                                  LossOnlyGlobals &) {}
};

struct TtqTraits : plastix::DefaultNetworkTraits<LossOnlyGlobals> {
  using ForwardPass = LinearForward;
  using BackwardPass = LinearBackward;
  using Loss = plastix::MSELoss;
  using UpdateConn = SGDUpdateConn;
  using ExtraUnitFields =
      plastix::UnitFieldList<plastix::alloc::SOAField<GradPreActTag, float>>;
  static constexpr size_t UnitCapacity = 64 * 1024;
  static constexpr size_t ConnCapacity = 16 * 1024 * 1024;
};

using TtqNet = plastix::Network<TtqTraits>;

// ---------------------------------------------------------------------------
// Layer builder. Same deterministic mask as the other benches.
// ---------------------------------------------------------------------------

struct SparseLinearLayer {
  size_t NumUnits;
  float Density;

  template <typename UA, typename CA>
  plastix::UnitRange operator()(UA &U, CA &C, plastix::UnitRange Prev) const {
    uint16_t SrcLevel = plastix::GetLevel(U, Prev.Begin);
    uint16_t NewLevel = SrcLevel + 1;
    plastix::UnitRange Units = U.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      plastix::GetLevel(U, Id) = NewLevel;

    size_t In = Prev.End - Prev.Begin;
    size_t Out = NumUnits;
    for (size_t Src = 0; Src < In; ++Src) {
      for (size_t Dst = 0; Dst < Out; ++Dst) {
        if (!EdgeExists(Src, Dst, Out, Density))
          continue;
        auto Cid = C.Allocate();
        plastix::GetField<plastix::FromIdTag>(C, Cid) =
            static_cast<uint32_t>(Prev.Begin + Src);
        plastix::GetField<plastix::ToIdTag>(C, Cid) =
            static_cast<uint32_t>(Units.Begin + Dst);
        plastix::GetField<plastix::SrcLevelTag>(C, Cid) = SrcLevel;
        plastix::GetWeight(C, Cid) = 0.1f * EdgeWeight(Src, Dst, Out);
      }
    }
    return Units;
  }
};

// ---------------------------------------------------------------------------
// Task definition. Holds dimensions, target generator, and a "tick"
// function that advances any time-varying state.
// ---------------------------------------------------------------------------

struct Task {
  std::string Name;
  size_t InDim;
  size_t HiddenDim;
  size_t OutDim;
  float Density;
  std::vector<float> (*MakeInput)(uint64_t Seed);
  void (*WriteTarget)(std::span<const float>, std::span<float>, void *);
  void (*TickState)(void *);
  void *State;
};

std::vector<float> MakeInputStatic(uint64_t S) {
  return RandomVector(8, S * 7 + 11);
}
void TargetStaticAdapter(std::span<const float> X, std::span<float> Y, void *) {
  TargetStatic(X, Y);
}
void TickNoop(void *) {}

std::vector<float> MakeInputIdempotent(uint64_t S) {
  return RandomVector(16, S * 11 + 13);
}
void TargetIdempotentAdapter(std::span<const float> X, std::span<float> Y,
                             void *) {
  TargetIdempotent(X, Y);
}

std::vector<float> MakeInputContinuousSmall(uint64_t S) {
  return RandomVector(4, S * 13 + 17);
}
void TargetContinuousSmallAdapter(std::span<const float> X, std::span<float> Y,
                                  void *St) {
  auto *S = static_cast<ContinuousSmallState *>(St);
  TargetContinuousSmall(X, Y, *S);
}
void TickContinuousSmall(void *St) {
  static_cast<ContinuousSmallState *>(St)->Step();
}

std::vector<float> MakeInputContinuousLarge(uint64_t S) {
  return RandomVector(32, S * 17 + 19);
}
void TargetContinuousLargeAdapter(std::span<const float> X, std::span<float> Y,
                                  void *St) {
  auto *S = static_cast<ContinuousLargeState *>(St);
  float Sum = 0.0f;
  for (auto V : X)
    Sum += V;
  Y[0] = S->Step(Sum);
}

// ---------------------------------------------------------------------------
// Plastix trainer. Loops DoStep until wall-clock budget elapses,
// writing a checkpoint every `SamplePeriodMs` ms.
// ---------------------------------------------------------------------------

struct Checkpoint {
  double ElapsedMs;
  double MseTrain;
  double MseHoldout;
  long long StepsCompleted;
};

// Holdout set: a fixed, never-trained-on bank of (x, y) pairs we
// re-evaluate at each checkpoint to track generalization.
struct Holdout {
  std::vector<std::vector<float>> Xs;
  std::vector<std::vector<float>> Ys;
};

Holdout BuildHoldout(const Task &T, size_t N) {
  Holdout H;
  H.Xs.reserve(N);
  H.Ys.reserve(N);
  // Snapshot any tick state so the holdout targets are stable.
  for (size_t I = 0; I < N; ++I) {
    auto X = T.MakeInput(1000000ULL + I);
    std::vector<float> Y(T.OutDim, 0.0f);
    T.WriteTarget(X, Y, T.State);
    H.Xs.push_back(std::move(X));
    H.Ys.push_back(std::move(Y));
  }
  return H;
}

double MeanSquaredError(std::span<const float> Pred,
                        std::span<const float> Target) {
  double Acc = 0.0;
  size_t N = std::min(Pred.size(), Target.size());
  for (size_t I = 0; I < N; ++I) {
    double D = Pred[I] - Target[I];
    Acc += D * D;
  }
  return (N == 0) ? 0.0 : Acc / N;
}

std::vector<Checkpoint> TrainPlastix(const Task &T, const Holdout &H,
                                     double BudgetMs, double SamplePeriodMs) {
  auto Net = std::make_unique<TtqNet>(
      T.InDim,
      SparseLinearLayer{T.HiddenDim, T.Density},
      SparseLinearLayer{T.OutDim, 1.0f});

  std::vector<Checkpoint> Out;
  auto Start = Clock::now();
  auto NextSample = Start + std::chrono::milliseconds(
                                static_cast<int64_t>(SamplePeriodMs));
  long long Steps = 0;
  double RollingMseTrain = 0.0;
  size_t RollingN = 0;

  uint64_t Seed = 1;
  std::vector<float> TgtBuf(T.OutDim, 0.0f);
  while (true) {
    auto Now = Clock::now();
    double Elapsed = std::chrono::duration<double, std::milli>(Now - Start)
                         .count();
    if (Elapsed >= BudgetMs)
      break;

    auto X = T.MakeInput(Seed++);
    T.WriteTarget(X, TgtBuf, T.State);
    Net->DoStep(std::span<const float>(X), std::span<const float>(TgtBuf));
    T.TickState(T.State);
    ++Steps;

    auto Pred = Net->GetOutput();
    RollingMseTrain += MeanSquaredError(Pred, TgtBuf);
    ++RollingN;

    if (Now >= NextSample) {
      double MseHold = 0.0;
      // We need to re-forward holdout examples without training on them.
      // DoForwardPass is non-mutating to weights.
      for (size_t I = 0; I < H.Xs.size(); ++I) {
        Net->DoForwardPass(std::span<const float>(H.Xs[I]));
        auto P = Net->GetOutput();
        MseHold += MeanSquaredError(P, H.Ys[I]);
      }
      MseHold /= std::max<size_t>(1, H.Xs.size());

      Checkpoint C;
      C.ElapsedMs = Elapsed;
      C.MseTrain = (RollingN > 0) ? (RollingMseTrain / RollingN) : 0.0;
      C.MseHoldout = MseHold;
      C.StepsCompleted = Steps;
      Out.push_back(C);
      RollingMseTrain = 0.0;
      RollingN = 0;
      NextSample += std::chrono::milliseconds(
          static_cast<int64_t>(SamplePeriodMs));
    }
  }
  return Out;
}

#ifdef PLASTIX_HAVE_TORCH

// Torch counterpart: nn::Sequential with the same In/Hidden/Out shape.
// One MSE-loss + SGD step per example so the wall-clock comparison is
// apples to apples (Plastix DoStep is also batch=1).
std::vector<Checkpoint> TrainTorch(const Task &T, const Holdout &H,
                                   double BudgetMs, double SamplePeriodMs) {
  torch::set_num_threads(1);
  torch::manual_seed(42);
  torch::NoGradGuard NoGradGlobal; // disabled in TrainStep below

  auto Net = torch::nn::Sequential(
      torch::nn::Linear(static_cast<int64_t>(T.InDim),
                        static_cast<int64_t>(T.HiddenDim)),
      torch::nn::Linear(static_cast<int64_t>(T.HiddenDim),
                        static_cast<int64_t>(T.OutDim)));
  torch::optim::SGD Opt(Net->parameters(), torch::optim::SGDOptions(0.01));

  std::vector<Checkpoint> Out;
  auto Start = Clock::now();
  auto NextSample = Start + std::chrono::milliseconds(
                                static_cast<int64_t>(SamplePeriodMs));
  long long Steps = 0;
  double RollingMseTrain = 0.0;
  size_t RollingN = 0;

  uint64_t Seed = 1;
  std::vector<float> TgtBuf(T.OutDim, 0.0f);
  while (true) {
    auto Now = Clock::now();
    double Elapsed = std::chrono::duration<double, std::milli>(Now - Start)
                         .count();
    if (Elapsed >= BudgetMs)
      break;

    auto X = T.MakeInput(Seed++);
    T.WriteTarget(X, TgtBuf, T.State);
    T.TickState(T.State);
    ++Steps;

    {
      torch::AutoGradMode AG(true);
      auto XT = torch::from_blob(X.data(),
                                 {1, static_cast<int64_t>(T.InDim)},
                                 torch::kFloat32).clone();
      auto YT = torch::from_blob(TgtBuf.data(),
                                 {1, static_cast<int64_t>(T.OutDim)},
                                 torch::kFloat32).clone();
      Opt.zero_grad();
      auto Pred = Net->forward(XT);
      auto Loss = torch::mse_loss(Pred, YT);
      Loss.backward();
      Opt.step();
      RollingMseTrain += Loss.template item<double>();
      ++RollingN;
    }

    if (Now >= NextSample) {
      double MseHold = 0.0;
      {
        torch::NoGradGuard NG;
        for (size_t I = 0; I < H.Xs.size(); ++I) {
          auto XT = torch::from_blob(const_cast<float *>(H.Xs[I].data()),
                                     {1, static_cast<int64_t>(T.InDim)},
                                     torch::kFloat32);
          auto YT = torch::from_blob(const_cast<float *>(H.Ys[I].data()),
                                     {1, static_cast<int64_t>(T.OutDim)},
                                     torch::kFloat32);
          auto Pred = Net->forward(XT);
          MseHold += torch::mse_loss(Pred, YT).template item<double>();
        }
      }
      MseHold /= std::max<size_t>(1, H.Xs.size());

      Checkpoint C;
      C.ElapsedMs = Elapsed;
      C.MseTrain = (RollingN > 0) ? (RollingMseTrain / RollingN) : 0.0;
      C.MseHoldout = MseHold;
      C.StepsCompleted = Steps;
      Out.push_back(C);
      RollingMseTrain = 0.0;
      RollingN = 0;
      NextSample += std::chrono::milliseconds(
          static_cast<int64_t>(SamplePeriodMs));
    }
  }
  return Out;
}

#endif // PLASTIX_HAVE_TORCH

// ---------------------------------------------------------------------------
// Validation. Closed-form checks on every target so a later edit to
// TargetXxx can't silently change the data the bench trains on.
// ---------------------------------------------------------------------------

void Die(const char *Msg) {
  std::fprintf(stderr, "time_to_quality validation failed: %s\n", Msg);
  std::exit(1);
}

bool ApproxEq(float A, float B, float Tol = 1e-5f) {
  return std::fabs(A - B) <= Tol;
}

void Validate() {
  // Static: x = [1,0,...,0] should give y = 0.5
  {
    std::vector<float> X(8, 0.0f);
    X[0] = 1.0f;
    std::vector<float> Y(1, 0.0f);
    TargetStatic(X, Y);
    if (!ApproxEq(Y[0], 0.5f))
      Die("TargetStatic([1,0,...]) != 0.5");
  }
  // Idempotent: y == x
  {
    std::vector<float> X = {1.0f, -2.0f, 3.5f, 0.0f};
    std::vector<float> Y(4, 99.0f);
    TargetIdempotent(X, Y);
    for (size_t I = 0; I < X.size(); ++I)
      if (!ApproxEq(Y[I], X[I]))
        Die("TargetIdempotent y != x");
  }
  // Continuous small: at phase=0, x=[0,...] -> sin(0) = 0
  {
    ContinuousSmallState S;
    std::vector<float> X(4, 0.0f);
    std::vector<float> Y(1, 0.0f);
    TargetContinuousSmall(X, Y, S);
    if (!ApproxEq(Y[0], 0.0f, 1e-6f))
      Die("TargetContinuousSmall(phase=0, x=0) != 0");
  }
  // Continuous large: first step with empty ring; XSum=0 -> 0.5*0 + tanh(0) = 0
  {
    ContinuousLargeState S;
    float Y = S.Step(0.0f);
    if (!ApproxEq(Y, 0.0f, 1e-6f))
      Die("ContinuousLarge first step != 0");
    // Second step: ring[0]=0 still (Cursor advanced), Y = 0.5*0 + tanh(0) = 0
    float Y2 = S.Step(0.0f);
    if (!ApproxEq(Y2, 0.0f, 1e-6f))
      Die("ContinuousLarge second-step-with-zero != 0");
  }
}

// ---------------------------------------------------------------------------
// Driver: run each (task, framework) and dump checkpoints to CSV.
// ---------------------------------------------------------------------------

void WriteRows(const std::filesystem::path &Csv, const Task &T,
               const std::string &Framework,
               const std::vector<Checkpoint> &Cps) {
  for (const auto &C : Cps) {
    CsvRow Row;
    Row.Set("task", T.Name);
    Row.Set("framework", Framework);
    Row.Set("in_dim", T.InDim);
    Row.Set("hidden_dim", T.HiddenDim);
    Row.Set("out_dim", T.OutDim);
    Row.Set("density", static_cast<double>(T.Density));
    Row.Set("elapsed_ms", C.ElapsedMs);
    Row.Set("steps", C.StepsCompleted);
    Row.Set("mse_train", C.MseTrain);
    Row.Set("mse_holdout", C.MseHoldout);
    AppendRow(Csv, Row);
  }
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
  Validate();

  const double BudgetMs = 3000.0;
  const double SampleMs = 100.0;

  ContinuousSmallState CsSmall;
  ContinuousLargeState CsLarge;

  std::vector<Task> Tasks = {
      {"static",            8,  32, 1, 0.5f,
       MakeInputStatic,           TargetStaticAdapter,           TickNoop,
       nullptr},
      {"idempotent",        16, 32, 16, 0.5f,
       MakeInputIdempotent,       TargetIdempotentAdapter,       TickNoop,
       nullptr},
      {"continuous_small",  4,  16, 1, 0.5f,
       MakeInputContinuousSmall,  TargetContinuousSmallAdapter,
       TickContinuousSmall,       &CsSmall},
      {"continuous_large",  32, 64, 1, 0.25f,
       MakeInputContinuousLarge,  TargetContinuousLargeAdapter,  TickNoop,
       &CsLarge},
  };

  auto Csv = OutputFile("time_to_quality.csv");
  std::filesystem::remove(Csv);

  for (auto &T : Tasks) {
    auto H = BuildHoldout(T, 64);
    auto Px = TrainPlastix(T, H, BudgetMs, SampleMs);
    WriteRows(Csv, T, "plastix", Px);
    std::fprintf(stderr, "[ttq] %-18s plastix: %zu checkpoints\n",
                 T.Name.c_str(), Px.size());

#ifdef PLASTIX_HAVE_TORCH
    auto Tx = TrainTorch(T, H, BudgetMs, SampleMs);
    WriteRows(Csv, T, "torch", Tx);
    std::fprintf(stderr, "[ttq] %-18s torch:   %zu checkpoints\n",
                 T.Name.c_str(), Tx.size());
#endif
  }

  std::fprintf(stderr, "[ttq] wrote %s\n", Csv.string().c_str());
  return 0;
}
