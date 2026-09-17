#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/conn.hpp"
#include "plastix/dispatch_cpu.hpp"
#include "plastix/dispatch_gpu.hpp"
#include "plastix/layers.hpp"
#include "plastix/reverse_adjacency.hpp"
#include "plastix/traits.hpp"
#include "plastix/unit_state.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "plastix/macros.hpp"

#ifdef PLASTIX_HAS_CUDA
#include "plastix/cuda_graph.hpp"
#include "plastix/device_alloc.hpp"
#include <cuda_runtime.h>
#endif

namespace plastix {

/// Returns the library version as a string.
const char *version();

// Network

static constexpr uint16_t MaxLevels = 1024;

// Compile-time backend witness: Do* phases pick cpu:: vs gpu:: via
// `if constexpr (HasCuda && ...)`; gpu side always declared for lookup.
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

struct ProposalTag {};
struct ProposalEntity {};

using ProposalScratchAllocator =
    alloc::SOAAllocator<ProposalEntity,
                        alloc::SOAField<ProposalTag, CompactEdge>>;

template <NetworkTraits Traits> class Network {
  using UnitAllocator = UnitAllocFor<Traits>;
  using ConnAllocator = ConnAllocFor<Traits>;
  using GlobalState = typename Traits::GlobalState;

  // Globals in unified memory (or heap in non-CUDA) so the single instance
  // is addressable from host and kernels, which take it by pointer.
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
      : NumInput(InputDim), UnitAlloc(Traits::UnitCapacity),
        ConnAlloc(Traits::ConnCapacity), Globals(AllocGlobals()),
        KahnAlloc(UnitAlloc.GetCapacity() + 1),
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
    if constexpr (Traits::ReverseAdjForward) {
#ifdef PLASTIX_HAS_CUDA
      PLASTIX_CUDA_CHECK(cudaMallocManaged(
          &RevOffsets, (UnitAlloc.GetCapacity() + 1) * sizeof(std::uint32_t)));
      PLASTIX_CUDA_CHECK(cudaMallocManaged(
          &RevIncoming, ConnAlloc.GetCapacity() * sizeof(std::uint32_t)));
#else
      RevOffsets = new std::uint32_t[UnitAlloc.GetCapacity() + 1];
      RevIncoming = new std::uint32_t[ConnAlloc.GetCapacity()];
#endif
      RebuildReverseAdjacency();
    }
    // Device-resident (ADR-006): upload the constructed staging mirror once.
    if constexpr (alloc::IsDeviceResidentV<UnitAllocator>) {
      UnitAlloc.PushToDevice();
      ConnAlloc.PushToDevice();
    }
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
    if constexpr (Traits::ReverseAdjForward) {
#ifdef PLASTIX_HAS_CUDA
      if (RevOffsets)
        PLASTIX_CUDA_CHECK(cudaFree(RevOffsets));
      if (RevIncoming)
        PLASTIX_CUDA_CHECK(cudaFree(RevIncoming));
#else
      delete[] RevOffsets;
      delete[] RevIncoming;
#endif
    }
#ifdef PLASTIX_HAS_CUDA
    if (CaptureStream_)
      PLASTIX_CUDA_CHECK(cudaStreamDestroy(CaptureStream_));
    if (DeviceTargets_)
      PLASTIX_CUDA_CHECK(cudaFree(DeviceTargets_));
#endif
  }

  // Rebuild the reverse-adjacency CSR (incoming edges grouped by destination
  // unit) used by the per-unit forward. Call after any structural change; the
  // ctor and DoAddConnections do this automatically when ReverseAdjForward.
  void RebuildReverseAdjacency() {
    if constexpr (Traits::ReverseAdjForward) {
      std::uint32_t *WritePos = KahnAlloc.template GetArrayFor<InDegreeTag>();
      BuildReverseAdjacency(ConnAlloc, UnitAlloc.Size(), RevOffsets,
                            RevIncoming, WritePos);
    }
  }

  Network(const Network &) = delete;
  Network &operator=(const Network &) = delete;
  Network(Network &&) = delete;
  Network &operator=(Network &&) = delete;

  size_t GetStep() const { return Step; }

  void DoForwardPass(std::span<const float> Inputs) {
    for (size_t I = 0; I < NumInput; ++I)
      GetActivation(UnitAlloc, I) = Inputs[I];
    // Device-resident: push just the input slice (kernels rewrite the rest).
    if constexpr (alloc::IsDeviceResidentV<UnitAllocator>)
      UnitAlloc.template PushFieldSlice<ActivationTag>(0, NumInput);
    RunForward();

    ++Step;
  }

#ifdef PLASTIX_HAS_CUDA
  // Device-input forward (ADR-007 M2): D2D-copy the input slice from a device
  // pointer (e.g. a torch CUDA tensor) into the arena, then run -- no host trip.
  void DoForwardPassDevice(const float *DeviceInput) {
    static_assert(alloc::IsDeviceResidentV<UnitAllocator>,
                  "DoForwardPassDevice requires the device-resident backend");
    float *Act = UnitAlloc.view().template GetArrayFor<ActivationTag>();
    PLASTIX_CUDA_CHECK(cudaMemcpy(Act, DeviceInput, NumInput * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
    RunForward();
    ++Step;
  }

  // Device-target loss: targets already on the device (no H2D upload).
  void DoCalculateLossDevice(const float *DeviceTargets) {
    if constexpr (!std::is_same_v<typename Traits::Loss, NoLoss>) {
      static_assert(alloc::IsDeviceResidentV<UnitAllocator>,
                    "DoCalculateLossDevice requires the device-resident backend");
      gpu::DoCalculateLossOnDevice<typename Traits::Loss>(
          UnitAlloc.view(), OutputRange, DeviceTargets, OutputRange.Size(),
          Globals);
    }
  }

  // Device pointer to the output activations (for zero-copy export to torch).
  const float *GetOutputDevice() const {
    static_assert(alloc::IsDeviceResidentV<UnitAllocator>,
                  "GetOutputDevice requires the device-resident backend");
    return UnitAlloc.view().template GetArrayFor<ActivationTag>() +
           OutputRange.Begin;
  }
#endif

  void DoCalculateLoss(std::span<const float> Targets) {
    if constexpr (std::is_same_v<typename Traits::Loss, NoLoss>)
      return;
    else {
      if (Targets.empty())
        return;
      // Device-resident: upload the target slice, run the loss on the device
      // view (no output D2H, no gradient H2D). #ifdef so the CPU build never
      // names cuda* / the device buffer.
#ifdef PLASTIX_HAS_CUDA
      if constexpr (alloc::IsDeviceResidentV<UnitAllocator>) {
        const size_t N = Targets.size();
        if (!DeviceTargets_)
          PLASTIX_CUDA_CHECK(cudaMalloc(
              reinterpret_cast<void **>(&DeviceTargets_),
              OutputRange.Size() * sizeof(float)));
        PLASTIX_CUDA_CHECK(cudaMemcpy(DeviceTargets_, Targets.data(),
                                      N * sizeof(float),
                                      cudaMemcpyHostToDevice));
        gpu::DoCalculateLossOnDevice<typename Traits::Loss>(
            UnitAlloc.view(), OutputRange, DeviceTargets_, N, Globals);
        return;
      }
#endif
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
    else if constexpr (HasCuda && alloc::IsDeviceResidentV<UnitAllocator>)
      // Keep Globals device-resident: reset it on the device instead of
      // dereferencing it on the host every step (avoids per-step migration).
      gpu::DoResetGlobalOnDevice<typename Traits::ResetGlobal>(Globals);
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

  // Reclaim slots in ConnAlloc held by tombstoned (DeadTag=true) entries.
  // (unit compaction not yet implemented)
  void DoCompactConnections() {
    if constexpr (!NetworkShrinks<Traits> || !HasConnAdd<Traits>)
      return;
    else {
      cpu::DoCompactConnections(ConnAlloc);
      // Compaction preserves the relative order of live connections
      if constexpr (Traits::Model == Propagation::Topological)
        NeedsResort = true;
    }
  }

  void DoAddUnits() {
    if constexpr (std::is_same_v<typename Traits::AddUnit, NoAddUnit>)
      return;
    else {
      using AP = typename Traits::AddUnit;
      const size_t Before = UnitAlloc.Size();
      if constexpr (HasCuda && Traits::KernelizeAdd)
        gpu::DoAddUnits<AP>(UnitAlloc, Globals, MaxLevels);
      else
        cpu::DoAddUnits<AP>(UnitAlloc, Globals, MaxLevels);
      UnitsAddedLastStep = UnitAlloc.Size() - Before;
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
      if (Committed) {
        NeedsResort = true;
        if constexpr (Traits::ReverseAdjForward)
          RebuildReverseAdjacency(); // keep the CSR current after growth
      }
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
    DoCompactConnections();
    DoAddUnits();
    // Skip the O(N^2) proposal sweep when this step added no units and no
    // densifier is active -- new edges only appear at freshly generated units
    // (unit-driven growth), or under bounded-fanout densification exposed by
    // GlobalState::GrowFanout. A direct DoAddConnections() call outside DoStep
    // always sweeps: an explicit caller has, by construction, a reason to.
    if constexpr (!std::is_same_v<typename Traits::AddConn, NoAddConn>) {
      bool Densify = false;
      if constexpr (requires { Globals->GrowFanout; })
        Densify = Globals->GrowFanout > 0;
      if (UnitsAddedLastStep > 0 || Densify)
        DoAddConnections();
    }
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

  // Structural accessors (needed by the Python binding / device data path).
  UnitRange GetOutputRange() const { return OutputRange; }
  size_t GetNumInput() const { return NumInput; }

  // Device-resident: refresh the output slice into host staging so GetOutput()
  // is current (no-op on managed). Call before GetOutput on a loss-free path.
  void SyncOutputToHost() {
    if constexpr (alloc::IsDeviceResidentV<UnitAllocator>)
      UnitAlloc.template PullFieldSlice<ActivationTag>(OutputRange.Begin,
                                                       OutputRange.Size());
  }

  auto &GetConnAlloc() { return ConnAlloc; }
  auto &GetUnitAlloc() { return UnitAlloc; }

  // Host access to the single live GlobalState (runtime policy params, staged
  // between phases). Managed/heap-backed, so directly host-addressable; a CUDA
  // caller must order its write before the kernels that read it.
  GlobalState &Global() { return *Globals; }
  const GlobalState &Global() const { return *Globals; }

private:
#ifdef PLASTIX_HAS_CUDA
  // Capture the forward sequence once (re-capture on a unit-count change) and
  // replay it (ADR-008 P0). RunFwd(stream, sync=false) issues it during capture.
  template <class RunFwd> void LaunchForwardGraph(RunFwd &&Run) {
    if (!CaptureStream_)
      PLASTIX_CUDA_CHECK(cudaStreamCreate(&CaptureStream_));
    if (!ForwardGraph_.ready() || ForwardGraphSize_ != UnitAlloc.Size()) {
      ForwardGraph_.Capture(CaptureStream_,
                            [&] { Run(CaptureStream_, /*Sync=*/false); });
      ForwardGraphSize_ = UnitAlloc.Size();
    }
    ForwardGraph_.Launch(CaptureStream_);
    PLASTIX_CUDA_CHECK(cudaStreamSynchronize(CaptureStream_));
  }
#endif

  // Forward dispatch shared by the host- and device-input entry points. The
  // graph path is entirely under #ifdef so the CPU build names no graph members.
  void RunForward() {
    using FP = typename Traits::ForwardPass;
    using Acc = typename FP::Accumulator;
#ifdef PLASTIX_HAS_CUDA
    constexpr bool UseGraph = std::is_same_v<Acc, float> &&
                              WantsCudaGraph<Traits>() &&
                              alloc::IsDeviceResidentV<UnitAllocator>;
    if constexpr (UseGraph) {
      if constexpr (Traits::Model == Propagation::Topological) {
        if (NeedsResort) {
          SortConnectionsByLevel();
          ForwardGraph_.Reset(); // level ranges changed; re-capture
        }
        LaunchForwardGraph([&](void *S, bool Sync) {
          gpu::DoForwardTopological<FP>(UnitAlloc, ConnAlloc, Globals, NumInput,
                                        Ranges, NumLevels, S, Sync);
        });
      } else if constexpr (Traits::ReverseAdjForward) {
        LaunchForwardGraph([&](void *S, bool Sync) {
          gpu::DoForwardReverseAdj<FP>(UnitAlloc, ConnAlloc, Globals, NumInput,
                                       RevOffsets, RevIncoming, S, Sync);
        });
      } else {
        LaunchForwardGraph([&](void *S, bool Sync) {
          gpu::DoForwardPipeline<FP>(UnitAlloc, ConnAlloc, Globals, NumInput, S,
                                     Sync);
        });
      }
      return;
    }
#endif
    // Eager GPU or CPU path.
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
      if constexpr (HasCuda && std::is_same_v<Acc, float> &&
                    Traits::ReverseAdjForward)
        gpu::DoForwardReverseAdj<FP>(UnitAlloc, ConnAlloc, Globals, NumInput,
                                     RevOffsets, RevIncoming);
      else if constexpr (HasCuda && std::is_same_v<Acc, float>)
        gpu::DoForwardPipeline<FP>(UnitAlloc, ConnAlloc, Globals, NumInput);
      else
        cpu::DoForwardPipeline<FP>(UnitAlloc, ConnAlloc, Globals, NumInput);
    }
  }

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
  // Units added by the most recent DoAddUnits(); gates the add-connection
  // proposal sweep (see DoAddConnections).
  size_t UnitsAddedLastStep = 0;
  KahnScratchAllocator KahnAlloc;
  ProposalScratchAllocator ProposalAlloc;
  // Reverse-adjacency CSR (managed) for the per-unit forward; allocated only
  // when Traits::ReverseAdjForward.
  std::uint32_t *RevOffsets = nullptr;
  std::uint32_t *RevIncoming = nullptr;

#ifdef PLASTIX_HAS_CUDA
  // Opt-in forward-pass CUDA graph (ADR-008 P0). Captured lazily on the first
  // forward and replayed while the unit count is unchanged.
  gpu::CudaGraph ForwardGraph_;
  cudaStream_t CaptureStream_ = nullptr;
  size_t ForwardGraphSize_ = static_cast<size_t>(-1);
  // Device target buffer for the device-side loss (ADR-007/008).
  float *DeviceTargets_ = nullptr;
#endif
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
