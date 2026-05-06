#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/conn.hpp"
#include "plastix/dispatch_cpu.hpp"
#include "plastix/dispatch_gpu.hpp"
#include "plastix/layers.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "plastix/macros.hpp"

#ifdef PLASTIX_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace plastix {

/// Returns the library version as a string.
const char *version();

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

static constexpr uint16_t MaxLevels = 1024;

// Compile-time backend witness. Every Do* phase picks between
// `plastix::cpu::Foo(...)` and `plastix::gpu::Foo(...)` with
// `if constexpr (HasCuda && <per-phase predicate>)`. The gpu side is
// always declared (empty stubs in host builds) so the qualified-name
// lookup in the discarded branch resolves.
#ifdef PLASTIX_HAS_CUDA
inline constexpr bool HasCuda = true;
#else
inline constexpr bool HasCuda = false;
#endif

struct LevelRange {
  uint32_t Begin;
  uint32_t End;
};

struct InDegreeTag {};
struct OutOffsetTag {};
struct KahnWritePosTag {};
struct FrontierTag {};
struct NextFrontierTag {};
struct KahnScratchEntity {};

using KahnScratchAllocator =
    alloc::SOAAllocator<KahnScratchEntity,
                        alloc::SOAField<InDegreeTag, uint32_t>,
                        alloc::SOAField<OutOffsetTag, uint32_t>,
                        alloc::SOAField<KahnWritePosTag, uint32_t>,
                        alloc::SOAField<FrontierTag, uint32_t>,
                        alloc::SOAField<NextFrontierTag, uint32_t>>;

struct CompactEdge {
  uint64_t Bits;

  CompactEdge() : Bits(0) {}
  CompactEdge(uint32_t From, uint32_t To)
      : Bits(static_cast<uint64_t>(From) | (static_cast<uint64_t>(To) << 32)) {}

  uint32_t From() const { return static_cast<uint32_t>(Bits); }
  uint32_t To() const { return static_cast<uint32_t>(Bits >> 32); }
};

struct ProposalTag {};
struct ProposalEntity {};

using ProposalScratchAllocator =
    alloc::SOAAllocator<ProposalEntity,
                        alloc::SOAField<ProposalTag, CompactEdge>>;

template <NetworkTraits Traits> class Network {
  using UnitAllocator = UnitAllocFor<Traits>;
  using ConnAllocator = ConnAllocFor<Traits>;
  using GlobalState = typename Traits::GlobalState;

  // Allocate Globals in unified memory (or heap, in non-CUDA builds) so the
  // single live GlobalState instance is addressable from both host code and
  // kernels. Kernels take Globals by pointer and dereference inside.
  static GlobalState *AllocGlobals() {
#ifdef PLASTIX_HAS_CUDA
    void *P = nullptr;
    PLASTIX_CUDA_CHECK(cudaMallocManaged(&P, sizeof(GlobalState)));
#else
    void *P = ::operator new(sizeof(GlobalState));
#endif
    return new (P) GlobalState{};
  }

public:
  template <typename... Builders>
    requires(sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, Builders... Layers)
      : Network(InputDim, NoUnitInit{}, Layers...) {}

  template <typename InputInit, typename... Builders>
    requires(std::invocable<InputInit, UnitAllocator &, size_t> &&
             sizeof...(Builders) > 0 &&
             (LayerBuilder<Builders, UnitAllocator, ConnAllocator> && ...))
  Network(size_t InputDim, InputInit Init, Builders... Layers)
      : NumInput(InputDim), UnitAlloc(4096), ConnAlloc(4096 * 4),
        Globals(AllocGlobals()), KahnAlloc(UnitAlloc.GetCapacity() + 1),
        ProposalAlloc(ConnAlloc.GetCapacity()) {
    for (size_t I = 0; I < InputDim; ++I) {
      auto Id = UnitAlloc.Allocate();
      Init(UnitAlloc, Id);
    }
    UnitRange Prev{0, InputDim};
    ((Prev = Layers(UnitAlloc, ConnAlloc, Prev)), ...);
    OutputRange = Prev;
    if constexpr (Traits::Model == Propagation::Topological)
      SortConnectionsByLevel();
  }

  Network(size_t InputDim, size_t OutputDim = 1)
      : Network(InputDim, FullyConnected<>{OutputDim}) {}

  ~Network() {
    if (Globals) {
      Globals->~GlobalState();
#ifdef PLASTIX_HAS_CUDA
      PLASTIX_CUDA_CHECK(cudaFree(Globals));
#else
      ::operator delete(Globals);
#endif
    }
  }

  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;
  Network(Network &&) = delete;
  Network &operator=(Network &&) = delete;

  size_t GetStep() const { return Step; }

  void DoForwardPass(std::span<const float> Inputs) {
    using FP = typename Traits::ForwardPass;
    using Acc = typename FP::Accumulator;

    for (size_t I = 0; I < NumInput; ++I)
      GetActivation(UnitAlloc, I) = Inputs[I];

    // GPU kernels currently require Accumulator=float (atomicAdd<float> is
    // the only supported Combine). Non-float accumulators run the host loop.
    if constexpr (Traits::Model == Propagation::Topological) {
      if (NeedsResort)
        SortConnectionsByLevel();
      if constexpr (HasCuda && std::is_same_v<Acc, float>)
        gpu::DoForwardTopological<FP>(UnitAlloc, ConnAlloc, Globals, NumInput,
                                      Ranges, NumLevels);
      else
        cpu::DoForwardTopological<FP>(UnitAlloc, ConnAlloc, Globals, NumInput,
                                      Ranges, NumLevels);
    } else {
      if constexpr (HasCuda && std::is_same_v<Acc, float>)
        gpu::DoForwardPipeline<FP>(UnitAlloc, ConnAlloc, Globals, NumInput);
      else
        cpu::DoForwardPipeline<FP>(UnitAlloc, ConnAlloc, Globals, NumInput);
    }

    ++Step;
  }

  void DoCalculateLoss(std::span<const float> Targets) {
    if constexpr (std::is_same_v<typename Traits::Loss, NoLoss>)
      return;
    else {
      if (Targets.empty())
        return;
      Traits::Loss::CalculateLoss(UnitAlloc, OutputRange, Targets, *Globals);
    }
  }

  void DoBackwardPass() {
    if constexpr (std::is_same_v<typename Traits::BackwardPass, NoBackwardPass>)
      return;
    else {
      using BP = typename Traits::BackwardPass;
      using Acc = typename BP::Accumulator;

      if constexpr (Traits::Model == Propagation::Topological) {
        if constexpr (HasCuda && std::is_same_v<Acc, float>)
          gpu::DoBackwardTopological<BP>(UnitAlloc, ConnAlloc, Globals,
                                         NumInput, Ranges, NumLevels);
        else
          cpu::DoBackwardTopological<BP>(UnitAlloc, ConnAlloc, Globals,
                                         NumInput, Ranges, NumLevels);
      } else {
        if constexpr (HasCuda && std::is_same_v<Acc, float>)
          gpu::DoBackwardPipeline<BP>(UnitAlloc, ConnAlloc, Globals, NumInput);
        else
          cpu::DoBackwardPipeline<BP>(UnitAlloc, ConnAlloc, Globals, NumInput);
      }
    }
  }

  void DoResetGlobalState() {
    if constexpr (std::is_same_v<typename Traits::ResetGlobal,
                                 NoResetGlobalState>)
      return;
    else
      Traits::ResetGlobal::Reset(*Globals);
  }

  void DoUpdateUnitState() {
    if constexpr (std::is_same_v<typename Traits::UpdateUnit, NoUpdateUnit>)
      return;
    else {
      using UP = typename Traits::UpdateUnit;
      if constexpr (HasCuda && Traits::KernelizeUpdate)
        gpu::DoUpdateUnit<UP>(UnitAlloc, Globals);
      else
        cpu::DoUpdateUnit<UP>(UnitAlloc, Globals);
    }
  }

  void DoUpdateConnectionState() {
    if constexpr (std::is_same_v<typename Traits::UpdateConn, NoUpdateConn>)
      return;
    else {
      using UC = typename Traits::UpdateConn;
      if constexpr (HasCuda && Traits::KernelizeUpdate)
        gpu::DoUpdateConn<UC>(UnitAlloc, ConnAlloc, Globals);
      else
        cpu::DoUpdateConn<UC>(UnitAlloc, ConnAlloc, Globals);
    }
  }

  void DoPruneUnits() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>)
      return;
    else {
      using PP = typename Traits::PruneUnit;
      if constexpr (HasCuda && Traits::KernelizePrune)
        gpu::DoPruneUnits<PP>(UnitAlloc, Globals);
      else
        cpu::DoPruneUnits<PP>(UnitAlloc, Globals);
    }
  }

  void DoPruneConnections() {
    if constexpr (std::is_same_v<typename Traits::PruneUnit, NoPruneUnit> &&
                  std::is_same_v<typename Traits::PruneConn, NoPruneConn>)
      return;
    else {
      using CP = typename Traits::PruneConn;
      constexpr bool HasUnitPrune =
          !std::is_same_v<typename Traits::PruneUnit, NoPruneUnit>;
      constexpr bool HasConnPrune =
          !std::is_same_v<typename Traits::PruneConn, NoPruneConn>;

      if constexpr (HasCuda && Traits::KernelizePrune)
        gpu::DoPruneConnections<HasUnitPrune, HasConnPrune, CP>(
            UnitAlloc, ConnAlloc, Globals);
      else
        cpu::DoPruneConnections<HasUnitPrune, HasConnPrune, CP>(
            UnitAlloc, ConnAlloc, Globals);
    }
  }

  void DoAddUnits() {
    if constexpr (std::is_same_v<typename Traits::AddUnit, NoAddUnit>)
      return;
    else {
      using AP = typename Traits::AddUnit;
      if constexpr (HasCuda && Traits::KernelizeAdd)
        gpu::DoAddUnits<AP>(UnitAlloc, Globals, MaxLevels);
      else
        cpu::DoAddUnits<AP>(UnitAlloc, Globals, MaxLevels);
    }
  }
  void DoAddConnections() {
    if constexpr (std::is_same_v<typename Traits::AddConn, NoAddConn>)
      return;
    else {
      using AC = typename Traits::AddConn;
      constexpr uint16_t N = Traits::Neighbourhood;
      bool Committed;
      if constexpr (HasCuda && Traits::KernelizeAdd)
        Committed = gpu::DoAddConnections<AC>(UnitAlloc, ConnAlloc,
                                              ProposalAlloc, Globals, N);
      else
        Committed = cpu::DoAddConnections<AC>(UnitAlloc, ConnAlloc, KahnAlloc,
                                              ProposalAlloc, Globals, N);
      if (Committed)
        NeedsResort = true;
    }
  }

  void DoStep(std::span<const float> Inputs,
              std::span<const float> Targets = {}) {
    DoForwardPass(Inputs);
    DoCalculateLoss(Targets);
    DoBackwardPass();
    DoUpdateUnitState();
    DoUpdateConnectionState();
    DoPruneUnits();
    DoPruneConnections();
    DoAddUnits();
    DoAddConnections();
    if constexpr (Traits::Model == Propagation::Topological) {
      if (NeedsResort)
        SortConnectionsByLevel();
    }
    DoResetGlobalState();
  }

  std::span<const float> GetOutput() const {
    const float *Base = &GetActivation(UnitAlloc, OutputRange.Begin);
    return {Base, OutputRange.Size()};
  }

  auto &GetConnAlloc() { return ConnAlloc; }
  auto &GetUnitAlloc() { return UnitAlloc; }

private:
  void RecomputeLevels() {
    if constexpr (HasCuda)
      gpu::RecomputeLevels(UnitAlloc, ConnAlloc, KahnAlloc, NumInput);
    else
      cpu::RecomputeLevels(UnitAlloc, ConnAlloc, KahnAlloc, NumInput);
  }

  void SortConnectionsByLevel() {
    RecomputeLevels();

    size_t N = ConnAlloc.Size();
    if (N == 0) {
      NumLevels = 0;
      return;
    }

    for (size_t C = 0; C < N; ++C) {
      auto From = GetField<FromIdTag>(ConnAlloc, C);
      GetField<SrcLevelTag>(ConnAlloc, C) = GetLevel(UnitAlloc, From);
    }

    uint32_t Histogram[MaxLevels] = {};
    for (size_t C = 0; C < N; ++C)
      ++Histogram[GetField<SrcLevelTag>(ConnAlloc, C)];

    uint32_t Offset = 0;
    NumLevels = 0;
    for (uint32_t L = 0; L < MaxLevels; ++L) {
      Ranges[L].Begin = Offset;
      Offset += Histogram[L];
      Ranges[L].End = Offset;
      if (Histogram[L] > 0)
        NumLevels = L + 1;
    }

    uint32_t WritePos[MaxLevels];
    for (uint32_t L = 0; L < MaxLevels; ++L)
      WritePos[L] = Ranges[L].Begin;

    size_t *Perm = ConnAlloc.PermutationScratch();
    for (size_t C = 0; C < N; ++C) {
      uint16_t Lvl = GetField<SrcLevelTag>(ConnAlloc, C);
      Perm[WritePos[Lvl]++] = C;
    }

    ConnAlloc.Gather(N);
    NeedsResort = false;
  }

  size_t NumInput;
  size_t Step = 0;
  UnitRange OutputRange;
  UnitAllocator UnitAlloc;
  ConnAllocator ConnAlloc;
  GlobalState *Globals;
  std::array<LevelRange, MaxLevels> Ranges{};
  uint16_t NumLevels = 0;
  bool NeedsResort = false;
  KahnScratchAllocator KahnAlloc;
  ProposalScratchAllocator ProposalAlloc;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
