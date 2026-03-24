// ===========================================================================
// SwiftTD Imprinting Learner
// ===========================================================================
//
// Implements a continual learning system (Chapter 9 of the thesis) that
// simultaneously generates new features, learns TD predictions via SwiftTD,
// and can remove stale features.
//
// Feature types:
//   ObservationFeature — mirrors input, always tenured, never removed.
//   PatternFeature     — fires when >k0 fraction of connected features active.
//   MemoryFeature      — fires k2 steps after its connected feature fires.
//                        Uses a shift-register (PulseTag) for efficient tracking.
//
// Execution per step:
//   1. Observe: set observation features from input
//   2. Consume: update generated features (pattern/memory activation)
//   3. Generate: create new features if tau_t <= eta + alpha_init
//   4. Forward pass: compute prediction V = dot(w, phi)
//   5. SwiftTD Phase 1 (UpdateUnit): trace decay, weight delta, h advance
//   6. Compute Phase 2 globals: ZDotPhi, E, V_delta
//   7. SwiftTD Phase 2 (UpdateConn): trace increment, weight update
//   8. Advance V_old
// ===========================================================================

#include "plastix/alloc.hpp"
#include "plastix/unit_state.hpp"
#include <plastix/plastix.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <vector>

// ---------------------------------------------------------------------------
// SOA field tags
// ---------------------------------------------------------------------------

struct PulseTag {};
struct DelayTag {};
struct ActivationThresholdTag {};
struct EligibilityTraceTag {};
struct StepSizeTag {};
struct ValueNodeTag {};
struct WeightTag {};
struct TenureTag {};

struct BetaTag {};
struct PreviousHTag {};
struct TemporaryHTag {};
struct DeltaZTag {};
struct PTag {};
struct HTag {};
struct ZTag {};
struct ZBarTag {};
struct DeltaVTag {};
struct OldVTag {};

using ImprintingLearnerAllocator = plastix::alloc::SOAAllocator<
    plastix::UnitState,
    plastix::alloc::SOAField<plastix::ActivationATag, float>,
    plastix::alloc::SOAField<plastix::ActivationBTag, float>,
    plastix::alloc::SOAField<plastix::BackwardAccTag, float>,
    plastix::alloc::SOAField<plastix::UpdateAccTag, float>,
    plastix::alloc::SOAField<plastix::PrunedTag, bool>,
    plastix::alloc::SOAField<plastix::PositionXTag, float>,
    plastix::alloc::SOAField<plastix::PositionYTag, float>,
    plastix::alloc::SOAField<plastix::PositionZTag, float>,
    plastix::alloc::SOAField<plastix::RadiusOfInfluenceTag, float>,
    plastix::alloc::SOAField<PulseTag, uint16_t>,
    plastix::alloc::SOAField<DelayTag, uint8_t>,
    plastix::alloc::SOAField<ActivationThresholdTag, float>,
    plastix::alloc::SOAField<EligibilityTraceTag, float>,
    plastix::alloc::SOAField<ValueNodeTag, float>,
    plastix::alloc::SOAField<WeightTag, float>,
    plastix::alloc::SOAField<TenureTag, uint8_t>,
    plastix::alloc::SOAField<BetaTag, float>,
    plastix::alloc::SOAField<PreviousHTag, float>,
    plastix::alloc::SOAField<TemporaryHTag, float>,
    plastix::alloc::SOAField<DeltaZTag, float>,
    plastix::alloc::SOAField<PTag, float>,
    plastix::alloc::SOAField<HTag, float>,
    plastix::alloc::SOAField<ZTag, float>,
    plastix::alloc::SOAField<ZBarTag, float>,
    plastix::alloc::SOAField<DeltaVTag, float>,
    plastix::alloc::SOAField<OldVTag, float>>;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

struct ImprintingLearnerGlobalState {
  float Cumulant    = 0.0f;
  float V_old       = 0.0f;
  float V_delta     = 0.0f;
  float Delta       = 0.0f;
  float E           = 0.5f;
  float ZDotPhi     = 0.0f;

  float Eta           = 0.5f;
  float AlphaInit     = 1e-7f;
  float MetaLr        = 1e-3f;
  float NumericalEps  = 1e-8f;
  float Epsilon       = 0.9f;
  float Lambda        = 0.95f;
  float GammaDiscount = 0.1f;

  // Output unit range — CalculateAndApply only writes ValueNodeTag for units
  // in [OutputBegin, OutputEnd). Default covers all units for backward compat.
  size_t OutputBegin = 0;
  size_t OutputEnd   = static_cast<size_t>(-1);
};

// ---------------------------------------------------------------------------
// Forward pass — only writes ValueNodeTag for the output unit
// ---------------------------------------------------------------------------

struct ImprintingLearnerForwardPass {
  static float Map(auto &, size_t, auto &, float W, float A) {
    return W * A;
  }
  static float CalculateAndApply(auto &U, size_t Id, auto &G, float Mu) {
    if (Id >= G.OutputBegin && Id < G.OutputEnd)
      U.template Get<ValueNodeTag>(Id) = Mu;
    return Mu;
  }
};

// ---------------------------------------------------------------------------
// Phase 1 — UpdateUnit (trace decay, h advance, weight delta)
// ---------------------------------------------------------------------------

struct ImprintingLearnerUpdateUnit {
  using Partial = float;

  static Partial Map(auto &U, size_t, size_t SrcId, auto &G, float) {
    float z_old       = U.template Get<ZTag>(SrcId);
    float z_delta_old = U.template Get<DeltaZTag>(SrcId);
    float z_bar_old   = U.template Get<ZBarTag>(SrcId);
    float h_temp_old  = U.template Get<TemporaryHTag>(SrcId);
    float h_old       = U.template Get<HTag>(SrcId);
    float beta        = U.template Get<BetaTag>(SrcId);
    float alpha       = std::exp(beta);

    float dw = (z_old != 0.0f)
                   ? G.Delta * z_old - z_delta_old * G.V_delta
                   : 0.0f;
    U.template Get<WeightTag>(SrcId) = dw;

    if (z_old != 0.0f) {
      U.template Get<PreviousHTag>(SrcId) = h_old;
      U.template Get<HTag>(SrcId) = h_temp_old;
      U.template Get<TemporaryHTag>(SrcId) =
          h_temp_old + G.Delta * z_bar_old - z_delta_old * G.V_delta;

      beta += G.MetaLr / (alpha + G.NumericalEps);
      beta = std::min(beta, std::log(G.Eta));
      U.template Get<BetaTag>(SrcId) = beta;

      float decay = G.GammaDiscount * G.Lambda;
      U.template Get<ZTag>(SrcId)    = z_old * decay;
      U.template Get<PTag>(SrcId)   *= decay;
      U.template Get<ZBarTag>(SrcId) = z_bar_old * decay;
      U.template Get<DeltaZTag>(SrcId) = 0.0f;
    }
    return 0.0f;
  }

  static Partial Combine(Partial A, Partial B) { return A + B; }
  static void Apply(auto &, size_t, auto &, Partial) {}
};

// ---------------------------------------------------------------------------
// Phase 2 — UpdateConn (trace increment + weight apply)
// ---------------------------------------------------------------------------

struct ImprintingLearnerUpdateConn {
  static void UpdateIncomingConnection(auto &U, size_t, size_t SrcId,
                                       auto &C, size_t PageId, size_t SlotIdx,
                                       auto &G) {
    float phi = U.template Get<ValueNodeTag>(SrcId);

    if (phi != 0.0f) {
      float alpha = std::exp(U.template Get<BetaTag>(SrcId));
      float z_delta_new = (G.Eta / G.E) * alpha * phi;
      float z_now =
          U.template Get<ZTag>(SrcId) + z_delta_new * (1.0f - G.ZDotPhi);
      U.template Get<ZTag>(SrcId) = z_now;
      U.template Get<PTag>(SrcId) += phi * U.template Get<HTag>(SrcId);

      if (G.E <= G.Eta) {
        float z_bar = U.template Get<ZBarTag>(SrcId);
        U.template Get<ZBarTag>(SrcId) =
            z_bar + z_delta_new * (1.0f - G.ZDotPhi - phi * z_bar);
        float h_temp = U.template Get<TemporaryHTag>(SrcId);
        h_temp -= U.template Get<PreviousHTag>(SrcId) * phi *
                  (z_now - z_delta_new);
        h_temp -= U.template Get<HTag>(SrcId) * z_delta_new * phi;
        U.template Get<TemporaryHTag>(SrcId) = h_temp;
      } else {
        U.template Get<BetaTag>(SrcId) +=
            std::abs(phi) * std::log(G.Epsilon);
        U.template Get<TemporaryHTag>(SrcId) = 0.0f;
        U.template Get<HTag>(SrcId) = 0.0f;
        U.template Get<ZBarTag>(SrcId) = 0.0f;
      }
      U.template Get<DeltaZTag>(SrcId) = z_delta_new;
    }

    auto &Page = C.template Get<plastix::ConnPageMarker>(PageId);
    Page.Conn[SlotIdx].second += U.template Get<WeightTag>(SrcId);
  }

  static void UpdateOutgoingConnection(auto &, size_t, size_t, auto &, size_t,
                                       size_t, auto &) {}
};

// ---------------------------------------------------------------------------
// Backward pass (unused stub)
// ---------------------------------------------------------------------------

struct ImprintingLearnerBackwardPass {
  static float Map(auto &, size_t, auto &, float, float) { return 0.0f; }
  static float CalculateAndApply(auto &, size_t, auto &, float) {
    return 0.0f;
  }
};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

struct ImprintingLearnerTraits
    : plastix::DefaultNetworkTraits<ImprintingLearnerAllocator,
                                    plastix::ConnStateAllocator,
                                    ImprintingLearnerGlobalState> {
  using ForwardPass  = ImprintingLearnerForwardPass;
  using BackwardPass = ImprintingLearnerBackwardPass;
  using UpdateUnit   = ImprintingLearnerUpdateUnit;
  using UpdateConn   = ImprintingLearnerUpdateConn;
};

using ILNet = plastix::Network<ImprintingLearnerTraits>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void InitFeatureUnit(ImprintingLearnerAllocator &U, size_t Id,
                            float AlphaInit) {
  U.Get<BetaTag>(Id) = std::log(AlphaInit);
}

static void InitObservationUnits(ImprintingLearnerAllocator &U, size_t Id) {
  U.Get<TenureTag>(Id) = 3; // TENURED
}

static void ComputePhase2Globals(ILNet &Net,
                                 ImprintingLearnerGlobalState &G,
                                 size_t) {
  auto &U = Net.GetUnitAlloc();
  auto &C = Net.GetConnAlloc();

  float ZDotPhi   = 0.0f;
  float Tau       = 0.0f;
  float VdeltaNew = 0.0f;

  for (size_t P = 0; P < C.Size(); ++P) {
    auto &Page = C.Get<plastix::ConnPageMarker>(P);
    for (size_t S = 0; S < Page.Count; ++S) {
      size_t Src = Page.Conn[S].first;
      float phi  = U.Get<ValueNodeTag>(Src);
      float z    = U.Get<ZTag>(Src);
      float alpha = std::exp(U.Get<BetaTag>(Src));
      float dw   = U.Get<WeightTag>(Src);
      ZDotPhi   += z * phi;
      Tau       += alpha * phi * phi;
      VdeltaNew += dw * phi;
    }
  }
  G.ZDotPhi = ZDotPhi;
  G.E       = std::max(G.Eta, Tau);
  G.V_delta = VdeltaNew;
}

// ---------------------------------------------------------------------------
// ImprintingLearner — wraps ILNet with feature generation and consume logic
// ---------------------------------------------------------------------------

class ImprintingLearner {
public:
  struct Config {
    size_t NumObs;
    size_t GenerationLimit = 5;
    bool GenPattern        = true;
    bool GenMemory         = true;
    float K0               = 0.9f;  // pattern threshold
    uint8_t MemDelay       = 1;     // memory delay (k2)
    float LrInit           = 1e-7f;
    float MetaLr           = 1e-3f;
    float Epsilon          = 0.9f;
    float Eta              = 0.5f;
    float Lambda           = 0.95f;
    float Gamma            = 0.1f;
    uint32_t Seed          = 42;
  };

  struct StepResult {
    float Prediction;
    float Delta;
  };

  explicit ImprintingLearner(Config C)
      : Net(C.NumObs, plastix::FullyConnected{1, 0.0f}),
        Cfg(C), Rng(C.Seed) {
    auto &U = Net.GetUnitAlloc();
    auto &G = Net.GetGlobalState();

    // Configure hyperparameters
    G.AlphaInit     = Cfg.LrInit;
    G.MetaLr        = Cfg.MetaLr;
    G.Epsilon       = Cfg.Epsilon;
    G.Eta           = Cfg.Eta;
    G.Lambda        = Cfg.Lambda;
    G.GammaDiscount = Cfg.Gamma;
    G.OutputBegin   = Cfg.NumObs;
    G.OutputEnd     = Cfg.NumObs + 1;

    // Initialize observation features
    for (size_t I = 0; I < Cfg.NumObs; ++I) {
      InitFeatureUnit(U, I, Cfg.LrInit);
      InitObservationUnits(U, I);
      Features.push_back({FT_Observation, 0, {}});
    }

    // Output unit metadata (index = NumObs)
    U.Get<TenureTag>(Cfg.NumObs) = 4;
    Features.push_back({FT_Output, 0, {}});
    FeatureCount = Cfg.NumObs;
  }

  StepResult Step(std::span<const float> Inputs, float Cumulant) {
    auto &U = Net.GetUnitAlloc();
    auto &G = Net.GetGlobalState();

    // 1. Observe
    for (size_t I = 0; I < Cfg.NumObs; ++I)
      U.Get<ValueNodeTag>(I) = Inputs[I];

    // 2. Consume generated features
    ConsumeFeatures();

    // 3. Generate new features
    if (Cfg.GenPattern || Cfg.GenMemory)
      GenerateFeatures();

    // 4. Copy phi into activation buffer
    CopyPhiToActivationBuffer();

    // 5. Forward pass
    Net.DoForwardPass(Inputs);
    float V = Net.GetOutput()[0];

    // 6. TD error
    G.Delta = Cumulant + G.GammaDiscount * V - G.V_old;

    // 7. Phase 1
    Net.DoUpdateUnitState();

    // 8. Phase 2 globals
    ComputePhase2Globals(Net, G, 0);

    // 9. Phase 2
    Net.DoUpdateConnectionState();

    // 10. Advance V_old
    G.V_old = V;

    return {V, G.Delta};
  }

  ILNet &GetNet() { return Net; }
  size_t NumTotalFeatures() const {
    return FeatureCount;
  }

private:
  enum FeatureType : uint8_t { FT_Observation, FT_Pattern, FT_Memory, FT_Output };

  struct FeatureMeta {
    FeatureType Type = FT_Observation;
    size_t ConnectedUnit = 0;          // for memory features
    std::vector<size_t> Sources;       // for pattern features
  };

  void ConsumeFeatures() {
    auto &U = Net.GetUnitAlloc();
    size_t FirstGen = Cfg.NumObs + 1;
    for (size_t I = FirstGen; I < U.Size(); ++I) {
      auto &Meta = Features[I];
      if (Meta.Type == FT_Memory) {
        auto &pulse = U.Get<PulseTag>(I);
        if (U.Get<ValueNodeTag>(Meta.ConnectedUnit) == 1.0f)
          pulse |= static_cast<uint16_t>(1u << U.Get<DelayTag>(I));
        U.Get<ValueNodeTag>(I) = (pulse & 1u) ? 1.0f : 0.0f;
        pulse >>= 1;
      } else if (Meta.Type == FT_Pattern) {
        size_t active = 0;
        for (size_t src : Meta.Sources) {
          if (U.Get<ValueNodeTag>(src) == 1.0f)
            active++;
        }
        float frac =
            static_cast<float>(active) /
            static_cast<float>(Meta.Sources.size());
        U.Get<ValueNodeTag>(I) =
            (frac > U.Get<ActivationThresholdTag>(I)) ? 1.0f : 0.0f;
      }
    }
  }

  void GenerateFeatures() {
    auto &U = Net.GetUnitAlloc();
    auto &G = Net.GetGlobalState();
    size_t OutputUnit = Cfg.NumObs;

    // Collect tenured features (snapshot before generation loop)
    std::vector<size_t> Tenured;
    for (size_t I = 0; I < U.Size(); ++I) {
      if (I == OutputUnit) continue;
      // All features are tenured (threshold=0), matching Python
      Tenured.push_back(I);
    }
    if (Tenured.empty()) return;

    // tau_t = sum of exp(beta[i]) for all features
    float TauT = 0.0f;
    for (size_t I = 0; I < U.Size(); ++I) {
      if (I == OutputUnit) continue;
      TauT += std::exp(U.Get<BetaTag>(I));
    }

    std::vector<FeatureType> Available;
    if (Cfg.GenMemory)  Available.push_back(FT_Memory);
    if (Cfg.GenPattern) Available.push_back(FT_Pattern);
    if (Available.empty()) return;

    std::uniform_int_distribution<size_t> TypeDist(0, Available.size() - 1);

    for (size_t K = 0; K < Cfg.GenerationLimit; ++K) {
      if (TauT > G.Eta + G.AlphaInit) break;

      FeatureType Type = Available[TypeDist(Rng)];
      size_t NewId = U.Allocate();
      if (NewId == static_cast<size_t>(-1)) break;

      InitFeatureUnit(U, NewId, G.AlphaInit);
      U.Get<TenureTag>(NewId) = 3; // TENURED

      FeatureMeta Meta;
      Meta.Type = Type;

      if (Type == FT_Memory) {
        std::uniform_int_distribution<size_t> SrcDist(0, Tenured.size() - 1);
        Meta.ConnectedUnit = Tenured[SrcDist(Rng)];
        U.Get<PulseTag>(NewId) = 0;
        U.Get<DelayTag>(NewId) = Cfg.MemDelay;

        // Consume immediately
        auto &pulse = U.Get<PulseTag>(NewId);
        if (U.Get<ValueNodeTag>(Meta.ConnectedUnit) == 1.0f)
          pulse |= static_cast<uint16_t>(1u << Cfg.MemDelay);
        U.Get<ValueNodeTag>(NewId) = (pulse & 1u) ? 1.0f : 0.0f;
        pulse >>= 1;
      } else {
        U.Get<ActivationThresholdTag>(NewId) = Cfg.K0;

        // Random subset of tenured features
        std::vector<size_t> Shuffled = Tenured;
        std::shuffle(Shuffled.begin(), Shuffled.end(), Rng);
        std::uniform_int_distribution<size_t> CountDist(1, Tenured.size());
        size_t Count = CountDist(Rng);
        Meta.Sources.assign(Shuffled.begin(),
                            Shuffled.begin() + static_cast<ptrdiff_t>(Count));

        // Consume immediately
        size_t active = 0;
        for (size_t src : Meta.Sources) {
          if (U.Get<ValueNodeTag>(src) == 1.0f)
            active++;
        }
        float frac = static_cast<float>(active) /
                     static_cast<float>(Meta.Sources.size());
        U.Get<ValueNodeTag>(NewId) = (frac > Cfg.K0) ? 1.0f : 0.0f;
      }

      Features.push_back(Meta);
      AddConnectionToOutput(NewId, 0.0f);
      ++FeatureCount;

      if (U.Get<ValueNodeTag>(NewId) == 1.0f)
        TauT += G.AlphaInit;
    }
  }

  void AddConnectionToOutput(size_t SrcId, float Weight) {
    auto &C = Net.GetConnAlloc();
    uint32_t OutputUnit = static_cast<uint32_t>(Cfg.NumObs);

    // Search backward for a page with room
    for (size_t P = C.Size(); P > 0; --P) {
      auto &Page = C.Get<plastix::ConnPageMarker>(P - 1);
      if (Page.ToUnitIdx == OutputUnit &&
          Page.Count < plastix::ConnPageSlotSize) {
        Page.Conn[Page.Count] = {static_cast<uint32_t>(SrcId), Weight};
        Page.Count++;
        return;
      }
    }
    // Allocate new page
    auto NewId = C.Allocate();
    auto &NewPage = C.Get<plastix::ConnPageMarker>(NewId);
    NewPage.ToUnitIdx = OutputUnit;
    NewPage.Count = 1;
    NewPage.Conn[0] = {static_cast<uint32_t>(SrcId), Weight};
  }

  void CopyPhiToActivationBuffer() {
    auto &U = Net.GetUnitAlloc();
    size_t StepVal = Net.GetStep();
    size_t FirstGen = Cfg.NumObs + 1;
    for (size_t I = FirstGen; I < U.Size(); ++I) {
      float phi = U.Get<ValueNodeTag>(I);
      if (StepVal % 2 == 0)
        U.Get<plastix::ActivationATag>(I) = phi;
      else
        U.Get<plastix::ActivationBTag>(I) = phi;
    }
  }

  ILNet Net;
  Config Cfg;
  std::mt19937 Rng;
  std::vector<FeatureMeta> Features; // indexed by unit ID
  size_t FeatureCount = 0;           // excludes output unit
};

// ---------------------------------------------------------------------------
// main() — guarded so tests can include this file without conflict
// ---------------------------------------------------------------------------

#ifndef IMPRINTING_LEARNER_TEST

int main() {
  std::cout << "Plastix SwiftTD Imprinting Learner\n";
  std::cout << "====================================\n";

  ImprintingLearner::Config Cfg;
  Cfg.NumObs  = 2;
  Cfg.Gamma   = 0.0f;
  Cfg.GenPattern = false;
  Cfg.GenMemory  = true;

  ImprintingLearner Learner(Cfg);

  std::mt19937 Rng(42);
  std::bernoulli_distribution Dist(0.5);

  constexpr size_t NSteps = 5000;
  std::cout << std::fixed << std::setprecision(6);

  for (size_t T = 0; T < NSteps; ++T) {
    float X[2] = {Dist(Rng) ? 1.0f : 0.0f, Dist(Rng) ? 1.0f : 0.0f};
    float Reward = X[0] - X[1];
    auto [V, Delta] = Learner.Step(X, Reward);

    if ((T + 1) % 1000 == 0)
      std::cout << "Step " << std::setw(5) << T + 1
                << "  V=" << V
                << "  features=" << Learner.NumTotalFeatures() << "\n";
  }

  return 0;
}

#endif // IMPRINTING_LEARNER_TEST
