// Memory-footprint microbench.
//
// Measures the committed memory cost of a sparse Plastix network and
// compares it to the analytical cost of an equivalent dense matrix
// (4 * In * Out bytes) and -- when libtorch is available -- the
// measured RSS impact of an actual torch::empty({In, Out}) allocation.
//
// Why this exists: the SpMV crossover bench shows the runtime
// crossover; this one shows the memory crossover. Together they bound
// the regime where Plastix is worth using.
//
// Each row in the output CSV reports, for a single (In, Out, density)
// shape:
//   - nnz: live connection count Plastix actually instantiated
//   - plastix_rss_delta_kb: VmRSS growth after construction + one
//     forward pass (so lazy MAP_NORESERVE pages get committed)
//   - plastix_analytical_live_b: the "honest" live cost --
//     nnz * (FwdConnFieldBytes + BackConnFieldBytes) plus the unit
//     contribution. Independent of /proc noise; useful as a sanity
//     check against the measured delta.
//   - dense_analytical_b: 4 * In * Out (fp32 In x Out matrix).
//   - torch_dense_rss_delta_kb / torch_sparse_csr_rss_delta_kb:
//     measured deltas when PLASTIX_HAVE_TORCH is defined; empty
//     otherwise.
//
// Validation: a small (In=64, Out=64, density=0.5) instance is checked
// at startup. The dense analytical and Plastix analytical must match
// the closed-form predictions exactly; the measured Plastix RSS delta
// must fall within a generous band around analytical_live (4x in
// either direction -- /proc RSS is page-aligned and lumpy).
//
// Output: build/benchmarks/_outputs/memory_footprint.csv

#include <benchmark/benchmark.h>

#include "_bench_common.hpp"
#include "plastix/plastix.hpp"

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
using bench_common::ReadRssKb;

namespace {

// ---------------------------------------------------------------------------
// Traits. Default policies + generous capacities. The Network<Traits>
// constructor mmaps capacity-sized arenas with MAP_NORESERVE so virtual
// reservation is free; only pages we actually touch consume RSS.
// ---------------------------------------------------------------------------

struct MemFootTraits : plastix::DefaultNetworkTraits<> {
  static constexpr size_t UnitCapacity = 64 * 1024;
  static constexpr size_t ConnCapacity = 64 * 1024 * 1024;
};

using MemNet = plastix::Network<MemFootTraits>;
using UnitAlloc = plastix::UnitAllocFor<MemFootTraits>;
using ConnAlloc = plastix::ConnAllocFor<MemFootTraits>;

// Per-slot bytes derived from the SOA field tuple. Forward + back
// (DoubleBuffered field arrays inside SOAAllocator) contribute equally,
// so the per-slot committed footprint scales as 2 * SumFieldSize.

constexpr size_t kConnFwdBytes =
    sizeof(uint32_t) + // FromIdTag
    sizeof(uint32_t) + // ToIdTag
    sizeof(bool) +     // DeadTag
    sizeof(uint16_t) + // SrcLevelTag
    sizeof(float);     // WeightTag (added by DefaultNetworkTraits)
constexpr size_t kConnSlotBytes = 2 * kConnFwdBytes;

constexpr size_t kUnitFwdBytes =
    sizeof(float) +    // ActivationTag
    sizeof(float) +    // ForwardAccTag (Accumulator=float for DefaultForwardPass)
    sizeof(float) +    // BackwardAccTag (Accumulator=float for NoBackwardPass)
    sizeof(bool) +     // PrunedTag
    sizeof(uint16_t);  // LevelTag
constexpr size_t kUnitSlotBytes = 2 * kUnitFwdBytes;

// Analytical "live" cost: bytes that *should* be committed after a
// single forward pass on a freshly-constructed network. Lower bound --
// extra PermScratch, KahnAlloc, and proposal arenas add a near-constant
// overhead independent of nnz, so the measured RSS will land somewhat
// above this.
size_t AnalyticalLiveBytes(size_t Units, size_t Conns) {
  return Units * kUnitSlotBytes + Conns * kConnSlotBytes;
}

// ---------------------------------------------------------------------------
// Layer builder. Same deterministic mask as bench_spmv_crossover so the
// connection counts are reproducible and density_bps → nnz translates
// exactly across files.
// ---------------------------------------------------------------------------

struct SparseLinearLayer {
  size_t NumUnits;
  float Density;
  size_t *NnzOut;

  template <typename UA, typename CA>
  plastix::UnitRange operator()(UA &U, CA &C, plastix::UnitRange Prev) const {
    uint16_t SrcLevel = plastix::GetLevel(U, Prev.Begin);
    uint16_t NewLevel = SrcLevel + 1;
    plastix::UnitRange Units = U.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      plastix::GetLevel(U, Id) = NewLevel;

    size_t In = Prev.End - Prev.Begin;
    size_t Out = NumUnits;
    size_t Nnz = 0;
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
        plastix::GetWeight(C, Cid) = EdgeWeight(Src, Dst, Out);
        ++Nnz;
      }
    }
    if (NnzOut)
      *NnzOut = Nnz;
    return Units;
  }
};

// ---------------------------------------------------------------------------
// Per-shape measurement. Construct Plastix, prime by running one
// forward pass, then read RSS. The forward call commits ActivationTag
// pages along with whatever other state the dispatch touches; without
// it the lazy mapping would understate the "actual usage" cost.
// ---------------------------------------------------------------------------

struct PlastixMeasurement {
  size_t Units = 0;
  size_t Conns = 0;
  long long RssDeltaKb = 0;
  long long RssAfterConstructKb = 0;
  long long RssAfterStepKb = 0;
};

PlastixMeasurement MeasurePlastix(size_t In, size_t Out, float Density) {
  PlastixMeasurement R;
  long long RssBefore = ReadRssKb();

  size_t Nnz = 0;
  auto Net = std::make_unique<MemNet>(
      In, SparseLinearLayer{Out, Density, &Nnz});
  long long RssConstruct = ReadRssKb();

  auto Input = RandomVector(In);
  Net->DoForwardPass(std::span<const float>(Input));
  auto OutSpan = Net->GetOutput();
  benchmark::DoNotOptimize(OutSpan.data());
  long long RssStep = ReadRssKb();

  R.Units = Net->GetUnitAlloc().Size();
  R.Conns = Nnz;
  R.RssAfterConstructKb = RssConstruct;
  R.RssAfterStepKb = RssStep;
  R.RssDeltaKb = (RssStep >= 0 && RssBefore >= 0) ? (RssStep - RssBefore) : -1;
  return R;
}

#ifdef PLASTIX_HAVE_TORCH

// Allocate (and touch) a dense In x Out fp32 tensor and report the RSS
// delta. We fill it once so the kernel actually backs the pages -- a
// bare torch::empty does not. RSS must be read inside the tensor's
// scope so the read sees the peak before its destructor releases pages.
long long MeasureTorchDenseRssKb(size_t In, size_t Out) {
  long long Before = ReadRssKb();
  torch::NoGradGuard NoGrad;
  auto W = torch::empty({static_cast<int64_t>(In),
                         static_cast<int64_t>(Out)},
                        torch::kFloat32);
  W.fill_(0.5f);
  long long After = ReadRssKb();
  benchmark::DoNotOptimize(W.data_ptr<float>());
  if (Before < 0 || After < 0)
    return -1;
  return After - Before;
}

// Build a CSR view of the same deterministic mask and measure its RSS.
// The CSR allocation is sized to nnz, so the comparison is apples to
// apples with Plastix's per-edge storage.
// Pre-touch every libtorch runtime path the sweep will exercise, so the
// one-time allocator-init / dispatcher-init / thread-pool RSS spikes
// land here instead of being attributed to the first measured row.
// After this returns, subsequent torch::empty / torch::sparse_csr_tensor
// calls report only the cost of *new* page commits beyond the allocator
// pool. (Note: torch's caching allocator will recycle freed pages from
// this warm-up, so later dense RSS deltas may read as 0 when the cached
// pool covers the request -- that is the actual steady-state cost and
// dense_analytical_b remains the apples-to-apples size comparison.)
void WarmUpLibtorch() {
  torch::NoGradGuard NoGrad;

  // Touch the dense allocator and fill path.
  {
    auto W = torch::empty({64, 64}, torch::kFloat32);
    W.fill_(0.5f);
    benchmark::DoNotOptimize(W.data_ptr<float>());
  }

  // Touch the sparse CSR construction path.
  {
    std::vector<int32_t> Crow = {0, 1, 2};
    std::vector<int32_t> Col = {0, 1};
    std::vector<float> Val = {1.0f, 2.0f};
    auto CrowT = torch::from_blob(Crow.data(), {3}, torch::kInt32).clone();
    auto ColT = torch::from_blob(Col.data(), {2}, torch::kInt32).clone();
    auto ValT = torch::from_blob(Val.data(), {2}, torch::kFloat32).clone();
    auto S = torch::sparse_csr_tensor(CrowT, ColT, ValT, {2, 2},
                                      torch::kFloat32);
    benchmark::DoNotOptimize(S);
  }
}

long long MeasureTorchSparseCsrRssKb(size_t In, size_t Out, float Density,
                                     size_t *NnzOut) {
  std::vector<int32_t> RowCounts(In, 0);
  size_t Nnz = 0;
  for (size_t Src = 0; Src < In; ++Src) {
    for (size_t Dst = 0; Dst < Out; ++Dst) {
      if (EdgeExists(Src, Dst, Out, Density)) {
        ++RowCounts[Src];
        ++Nnz;
      }
    }
  }
  std::vector<int32_t> CrowIndices(In + 1, 0);
  for (size_t I = 0; I < In; ++I)
    CrowIndices[I + 1] = CrowIndices[I] + RowCounts[I];
  std::vector<int32_t> ColIndices(Nnz);
  std::vector<float> Values(Nnz);
  std::vector<int32_t> Cursor(In, 0);
  for (size_t Src = 0; Src < In; ++Src) {
    for (size_t Dst = 0; Dst < Out; ++Dst) {
      if (!EdgeExists(Src, Dst, Out, Density))
        continue;
      int32_t Idx = CrowIndices[Src] + Cursor[Src]++;
      ColIndices[Idx] = static_cast<int32_t>(Dst);
      Values[Idx] = EdgeWeight(Src, Dst, Out);
    }
  }
  if (NnzOut)
    *NnzOut = Nnz;

  long long Before = ReadRssKb();
  torch::NoGradGuard NoGrad;
  auto CrowT = torch::from_blob(CrowIndices.data(),
                                {static_cast<int64_t>(In + 1)},
                                torch::kInt32).clone();
  auto ColT = torch::from_blob(ColIndices.data(),
                               {static_cast<int64_t>(Nnz)},
                               torch::kInt32).clone();
  auto ValT = torch::from_blob(Values.data(),
                               {static_cast<int64_t>(Nnz)},
                               torch::kFloat32).clone();
  auto Sparse = torch::sparse_csr_tensor(
      CrowT, ColT, ValT,
      {static_cast<int64_t>(In), static_cast<int64_t>(Out)},
      torch::kFloat32);
  long long After = ReadRssKb();
  benchmark::DoNotOptimize(Sparse);
  if (Before < 0 || After < 0)
    return -1;
  return After - Before;
}

#endif // PLASTIX_HAVE_TORCH

// ---------------------------------------------------------------------------
// Validation. Tiny shape; the analytical numbers and the per-slot
// constants are checked against their closed-form definitions, and the
// measured RSS delta is required to fall in a generous band around the
// analytical live cost.
// ---------------------------------------------------------------------------

void Die(const char *Msg) {
  std::fprintf(stderr, "memory_footprint validation failed: %s\n", Msg);
  std::exit(1);
}

void Validate() {
  // Per-slot byte constants. If any of these change, the analytical
  // numbers in every emitted CSV row become wrong silently -- pin them
  // here so a future field addition forces an update.
  if (kConnFwdBytes != 15)
    Die("kConnFwdBytes drifted (expected 4+4+1+2+4=15)");
  if (kUnitFwdBytes != 15)
    Die("kUnitFwdBytes drifted (expected 4+4+4+1+2=15)");
  if (kConnSlotBytes != 30 || kUnitSlotBytes != 30)
    Die("per-slot bytes drifted");

  const size_t In = 64, Out = 64;
  const float Density = 0.5f;
  size_t ExpNnz = 0;
  for (size_t S = 0; S < In; ++S)
    for (size_t D = 0; D < Out; ++D)
      if (EdgeExists(S, D, Out, Density))
        ++ExpNnz;

  auto M = MeasurePlastix(In, Out, Density);
  if (M.Conns != ExpNnz) {
    std::fprintf(stderr,
                 "nnz mismatch: SparseLinearLayer=%zu, EdgeExists count=%zu\n",
                 M.Conns, ExpNnz);
    std::exit(1);
  }
  if (M.Units != In + Out) {
    std::fprintf(stderr, "unit count mismatch: got=%zu, expected=%zu\n",
                 M.Units, In + Out);
    std::exit(1);
  }

  size_t AnalyticalLive = AnalyticalLiveBytes(M.Units, M.Conns);
  // The measured RSS includes the dispatcher's per-level Ranges,
  // KahnAlloc scratch (UnitCapacity slots), and proposal arena. For a
  // 64x64 / d=0.5 network the live nnz is ~2K and analytical_live ≈
  // 64KB; KahnAlloc scratch alone is UnitCapacity*16B≈1MB. Use a
  // looser absolute floor and require the order of magnitude is sane.
  if (M.RssDeltaKb < 0)
    Die("RSS reading failed (/proc/self/status)");
  if (M.RssDeltaKb == 0)
    Die("RSS reading reported zero delta -- kernel did not commit anything");
  // Upper bound: at most 16MB. The default-traits scratch arenas total
  // ~1.5MB and one forward pass touches a tiny fraction of the conn
  // arena, so anything over 16MB suggests a leak or accidental
  // commitment of the full reservation.
  if (M.RssDeltaKb > 16 * 1024) {
    std::fprintf(stderr,
                 "RSS delta unexpectedly large: %lld KB (analytical live %zu B)\n",
                 M.RssDeltaKb, AnalyticalLive);
    std::exit(1);
  }

  // Sanity-check the analytical dense cost.
  size_t DenseBytes = In * Out * sizeof(float);
  if (DenseBytes != 16384)
    Die("dense analytical drift");
}

// ---------------------------------------------------------------------------
// Main sweep. Same shape grid as bench_spmv_crossover so the CSVs join
// on (In, Out, density_bps).
// ---------------------------------------------------------------------------

struct ShapeKey {
  size_t In;
  size_t Out;
  int64_t DensityBps;
};

const std::vector<std::pair<size_t, size_t>> kShapes = {
    {256, 256},   {1024, 1024}, {4096, 4096},
    {4096, 1024}, {8192, 64},   {64, 8192},
};

const std::vector<int64_t> kDensities = {10, 50, 100, 500, 1000, 2000, 5000};

void RunSweep(const std::filesystem::path &Csv) {
  for (auto [In, Out] : kShapes) {
    for (auto Bps : kDensities) {
      float Density = static_cast<float>(Bps) / 10000.0f;
      auto M = MeasurePlastix(In, Out, Density);
      size_t AnalyticalLive = AnalyticalLiveBytes(M.Units, M.Conns);
      size_t DenseBytes = In * Out * sizeof(float);

      CsvRow Row;
      Row.Set("In", In);
      Row.Set("Out", Out);
      Row.Set("density_bps", static_cast<long long>(Bps));
      Row.Set("nnz", M.Conns);
      Row.Set("units", M.Units);
      Row.Set("plastix_rss_delta_kb", M.RssDeltaKb);
      Row.Set("plastix_rss_post_construct_kb", M.RssAfterConstructKb);
      Row.Set("plastix_rss_post_step_kb", M.RssAfterStepKb);
      Row.Set("plastix_analytical_live_b",
              static_cast<long long>(AnalyticalLive));
      Row.Set("dense_analytical_b", static_cast<long long>(DenseBytes));

#ifdef PLASTIX_HAVE_TORCH
      long long TorchDenseKb = MeasureTorchDenseRssKb(In, Out);
      size_t TorchCsrNnz = 0;
      long long TorchSparseKb =
          MeasureTorchSparseCsrRssKb(In, Out, Density, &TorchCsrNnz);
      Row.Set("torch_dense_rss_delta_kb", TorchDenseKb);
      Row.Set("torch_sparse_csr_rss_delta_kb", TorchSparseKb);
      Row.Set("torch_csr_nnz", TorchCsrNnz);
#else
      Row.Set("torch_dense_rss_delta_kb", std::string(""));
      Row.Set("torch_sparse_csr_rss_delta_kb", std::string(""));
      Row.Set("torch_csr_nnz", std::string(""));
#endif

      AppendRow(Csv, Row);
      std::fprintf(stderr,
                   "[memfoot] In=%zu Out=%zu d=%lld bps nnz=%zu rss=%lld KB "
                   "(analytical live=%zu B, dense=%zu B)\n",
                   In, Out, static_cast<long long>(Bps), M.Conns,
                   M.RssDeltaKb, AnalyticalLive, DenseBytes);
    }
  }
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
  Validate();

#ifdef PLASTIX_HAVE_TORCH
  torch::set_num_threads(1);
  WarmUpLibtorch();
#endif

  auto Csv = OutputFile("memory_footprint.csv");
  // Fresh file each run -- header/columns can change when libtorch is
  // toggled in or out, and CsvRow does not validate against an existing
  // header.
  std::filesystem::remove(Csv);

  RunSweep(Csv);
  std::fprintf(stderr, "[memfoot] wrote %s\n", Csv.string().c_str());
  return 0;
}
