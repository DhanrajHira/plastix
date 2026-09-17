# ADR-008: Batching a plastix-like (online, evolving-topology) network

Status: proposed (research/design; nothing implemented)
Related: ADR-006 (device-resident SOA), ADR-007 (Python bindings). Batching is
flagged in both as the main throughput limiter for GPU / Python-driven training.
Research: Fable-class agent, 2026-09-17. Sources cited inline.

## Context

plastix is **online / per-sample**: each `DoStep` processes one input and may
mutate the topology (prune, rewire, add). The per-sample loop pays a fixed
per-step cost (~14 kernel launches + ~8 device-wide syncs + 3 host/device bus
crossings) **independent of network size**, so at small/sparse scale the GPU is
launch-bound (measured earlier: GPU 28x slower than CPU at 288 conns; only wins
6.2x at ~800k candidates). Batching amortizes that floor and raises occupancy.
This ADR maps the design space and challenges.

## Facts today (the design must respect)

- **One sample per step.** `DoForwardPass(std::span<const float>)` writes
  `NumInput` floats into `ActivationTag[0..NumInput)` (`plastix.hpp:177-182`),
  pushes that slice H2D on the device-resident backend, runs the model. `DoStep`
  order: Forward, Loss, Backward, UpdateUnit, UpdateConn, PruneUnits, PruneConns,
  Compact, AddUnits, (AddConns), Sort, ResetGlobal.
- **Unit SOA** (`unit_state.hpp`): Activation:float, ForwardAcc, BackwardAcc,
  Pruned:bool, Level:uint16 + user extras (DeepR adds PreAct, GradPreAct, Age,
  Task, Kind, FanInTarget). **Conn SOA** (`conn.hpp`): FromId, ToId, Dead,
  SrcLevel + user extras (Weight, DeepR Theta/Sign/Layer, optim opt/m,v,t).
- **Allocator padding trap:** every field arena is `Capacity * MaxFieldSize()`
  bytes -- every column padded to the widest field (`alloc.hpp:222-248`,
  `device_alloc.hpp:44-46`). A naive `SOAField<ActTag, float[B]>` would inflate
  *every* column to `4B`. Batched columns must NOT go through this sizing.
- **Kernels** (`cuda_kernels.hpp`): per-level conn sweep does
  `atomicAdd(acc[dst], Map(...))`; per-unit apply filters by level, calls Apply,
  zeroes acc; backward mirrors into `BackwardAcc[src]`; a per-unit reverse-adj
  reduce is one block/unit with a shared-mem tree reduce. All launchers end each
  phase with `cudaDeviceSynchronize()` on the legacy stream.
- **The delta-rule gradient is never materialized:** `optim::detail::EdgeGradient
  = GradTag[dst] * Activation[src]` (`optim.hpp:131-134`); DeepR computes the
  same inline. This is the hinge of batched training (S4.6).
- **RNG is counter-based** (`random.hpp`): seeds fold `G.StepCounter`, bumped once
  per `DoStep`. Under batching "step" = "batch step", RNG contract unchanged.

Launch floor ≈ ~14 launches + ~8 syncs + 3 bus crossings/sample ⇒ ~100-250 µs/
sample regardless of net size ([CUDA launch overhead ~4-6µs](https://ai-infrastructure.net/cuda-graphs/)).

## Decision: model (A) shared topology, batched per-sample state

One graph (topology + parameters + level structure) shared by all B samples; only
activation-like unit state gets a batch dimension; every structural and parameter
mutation happens **once per batch** from batch-aggregated statistics.

**Why not (B) "batch of graphs"** (PyG/DGL block-diagonal disjoint union,
[PyG batching](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html),
[Fey & Lenssen 2019](https://arxiv.org/pdf/1903.02428)): that solves
*variable-size inputs*, which plastix doesn't have. plastix's topology is the
single object being learned; B disjoint copies = B independent training runs, not
one run B× faster, with B× the atomics and no coalescing gain.

**Hybrid that IS free: replica/seed batching.** A layer builder that instantiates
R disconnected network copies in one `Network` gives R independent runs per launch
with zero core change -- useful for multi-seed DeepR sweeps (Bellec frame DeepR as
MCMC over configurations) to fill an under-occupied GPU. It does not speed a single
run; only (A) does. Both compose (A inside each replica).

## Field layout: batched vs shared

| Field | Where | Batched? | Reason |
|---|---|---|---|
| Activation | unit | **yes** `[Cap][B]` | per-sample forward state; input becomes `B×NumInput` |
| ForwardAcc | unit | **yes** | scatter target of conn sweep, per sample |
| BackwardAcc | unit | **yes** | scatter target of backward sweep, per sample |
| PreAct, GradPreAct (DeepR) | unit | **yes** | per-sample, read by Apply + edge gradient |
| Pruned, Level, Age, Task, Kind, FanInTarget | unit | shared | topology / per-unit params, batch-invariant |
| FromId, ToId, Dead, SrcLevel | conn | shared | topology (frozen during a batch) |
| Weight, Theta, Sign, Layer, opt/m,v,t | conn | shared | params + optimizer state; one update/batch |
| AtomicCount, Ranges[], NumLevels, RevAdj CSR, Kahn/Proposal | net | shared | topology bookkeeping |
| GlobalState | net | shared, **batch-level semantics** | scalar accumulators in G become batch sums; document, don't duplicate |

**Rule: no connection column is ever batched.** Per-edge, per-sample quantities
(the edge gradient) are consumed on the fly inside the conn kernel, never stored
(S4.6). Keeps Gather/compaction/CSR untouched (S5).

**Layout: unit-major, batch-minor -- `X[unit*B + b]`, lane = `b`.** In the sweep
`acc[dst*B+b] += w*act[src*B+b]`, consecutive lanes take consecutive `b` for the
same conn ⇒ `act`/`acc` are 32 consecutive floats/warp (one 128B transaction
each; 32 atomics to 32 distinct addresses, no intra-warp conflict). This is the
GNN-SpMM layout (dense feature/batch dim innermost:
[GE-SpMM](https://arxiv.org/pdf/2007.03179),
[FeatGraph](https://arxiv.org/pdf/2008.11359)). Sample-major `X[b*Cap+unit]`
keeps the random gather/scatter and strides the per-b loop by `Cap*4` -- rejected.

**Allocator consequence:** add a **batched-column concept** (`SOAField<Tag, T,
Batched>` or a parallel `BatchedFields...` pack), arena `Capacity*MaxBatch*
sizeof(T)`, accessor `Ptr[Id*MaxBatch + b]`, kept **outside** `MaxFieldSize()`.
On the device-resident backend batched columns are **device-only** (no host
staging except the input push / output pull) -- do not make ADR-006's ~2x into
~2x·B.

## Semantic shift: online -> minibatch, per algorithm class

Shared weights ⇒ per-sample weight update inside a batch is impossible; update
becomes `Δw = f(mean_b g_b)` once/batch, and prune/rewire/add run once/batch on
batch-aggregated stats.

- **Backprop + optimizers** (`optim.hpp` Sgd/Momentum/RmsProp/Adam/AdamW): this
  *is* minibatch SGD; nothing lost. `EdgeGradient` -> `(1/B) Σ_b
  GradTag[dst,b]·Act[src,b]`. Adam's per-edge `opt/t` counts batch steps (= optax).
  Parity vectors are at B=1 (keep bit-identical at B=1, or add a plastax vmap mean).
- **DeepR** (Bellec et al. 2018): batching is *more* faithful -- the paper does
  backprop over a minibatch (MNIST B=10 SGD lr .05; TIMIT B=32 Adam), noise/
  dormancy/rewire once per iteration. Today's per-sample `DoStep` is the B=1
  corner; `T = ηα²/18` uses per-update η, nothing rescales. Caveat: per-unit age/
  k-rule counters now count batch steps; nb01-04 reference trajectories must be
  re-derived at the same B. ([Bellec et al., ICLR 2018](https://arxiv.org/abs/1711.05136))
- **SET / RigL / Top-KAST**: already batch/epoch-level. RigL fixes the mask within
  a step, updates every ΔT=100 steps from that batch's dense gradient (Alg. 1,
  [Evci et al. 2020](https://proceedings.mlr.press/v119/evci20a/evci20a.pdf)); SET
  rewires per epoch ([Mocanu 2018](https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks));
  Top-KAST re-selects top-K periodically ([Jayakumar 2020](https://proceedings.neurips.cc/paper/2020/hash/ee76626ee11ada502d5dbf1fb5aae4d2-Abstract.html)).
  RigL's absent-edge score `|GradPreAct[dst]·Act[src]|` -> batch mean of that
  product. **Batch-level structural update is the DST norm; plastix's per-sample
  update is the outlier.**
- **Where online semantics genuinely can't be preserved (keep B=1, opt-in):**
  (i) streaming rules whose next input depends on the just-updated weights
  (TD/SwiftTD, streaming iPC, imprinting, stream-AC(λ)); (ii) `Propagation::Pipeline`
  (temporal) batches only as B independent streams (vectorised envs), not
  consecutive time steps; (iii) per-sample `GlobalState` scalar accumulators
  silently become per-batch -- expose `G.BatchSize`; (iv) prune/add criteria keyed
  on one sample's activation must be rewritten against batch stats (mean, fraction
  active) the framework supplies.

## Batched kernel sketches

Notation: `Bmax` compile-time stride, `B ≤ Bmax` runtime; batched index `X[Id*Bmax+b]`.

- **Forward conn sweep (per level):** grid over `(conn, b)`, `b` innermost. For
  B≥32 a warp = one conn × 32 samples: `Map` reads W[I] (broadcast),
  Act[From*Bmax+b] (coalesced), `atomicAdd(&Acc[To*Bmax+b], V)` (32 distinct
  consecutive addrs). `WarpAtomicAddKeyed` no longer collapses within a warp
  (lanes differ by b) -- keep it only for B<32 (key = `To*Bmax+b`). Optional:
  register-block R samples/thread with float4 loads; sm_90 vector `atomicAdd(float4*)`.
- **Forward unit apply (per level):** grid `(unit, b)`, `Apply(U_b, Id, G,
  Acc[Id*Bmax+b])`, zero. Coalesced; the level filter reads a shared column
  (broadcast across b lanes).
- **Per-unit reverse-adj reduce:** one block/unit; threads `b = tid%B`,
  `kgroup = tid/B`; loop k over the CSR accumulating `W[c]*Act[From*Bmax+b]` in a
  register, then shared-mem reduce across kgroups only (for B=256 the reduce
  vanishes). Row-per-warp SpMM, contention-free -- best for high in-degree.
- **Backward:** mirror -- `atomicAdd(&BwdAcc[From*Bmax+b], W[I]*GradPreAct[To*Bmax+b])`,
  then `BP::Apply` per (unit,b) writes GradPreAct via the surrogate.
- **Loss:** targets as `B×Out` device buffer (fits ADR-007 `DoCalculateLossDevice`);
  one thread/sample (or per (group,b) for GroupedSoftmaxCE<10>) runs the existing
  `CalculateLoss` body on a sample view. Removes the per-step D2H/H2D pair in
  `deepr_policies.hpp`.
- **Weight-gradient reduction -- NO separate kernel:** since the gradient is the
  delta-rule product, `UpdateConnIncomingKernel` computes per edge `g = (1/B)
  Σ_b GradPreAct[To*Bmax+b]*Act[From*Bmax+b]` -- a per-thread dot product over two
  contiguous B-float rows (vectorizable, no atomics, no scratch). Provide
  `BatchEdgeGradient<GradTag>(U, Dst, Src)`, switch `optim::detail::EdgeGradient`
  to it. A separate batch-reduce is needed only for **per-unit** stats (mean|grad|,
  activity fraction) consumed by prune/add -- a trivial coalesced reduce over b.

## Dynamic topology under batching: rules

1. Topology + parameters **frozen** for forward+loss+backward of a batch (they
   already are: Dead/FromId/ToId/Weight/Level/Ranges/CSR read-only in kernels).
2. UpdateConn/UpdateUnit/Prune*/Compact/Add*/Rewire/Sort run **once per batch**
   after backward, in today's `DoStep` order, reading batch-aggregated quantities
   only. `DoStep(B×in, B×out)` keeps the order; only the first 3 phases reshape.
3. Sort/Gather/Compact permute *connection* columns only -- with no batched conn
   column they're untouched. **Pre-existing hazard to note:** SortConnectionsByLevel
   runs on the host view (stale staging on the device-resident backend); DeepR
   avoids it by never resorting -- any batched device-add path must fix this
   independently.
4. Unit compaction (unimplemented) must gather batched unit columns as `Bmax`-float
   rows (not stride 1) -- note for the future implementer.
5. Allocate/AllocateMany placement-new each field at Id (incl. device
   AddUnitsKernel); batched columns need their Bmax-slice zeroed at allocation
   (one-time; transient thereafter).
6. RevAdj CSR rebuild + Kahn RecomputeLevels are topology-only, unchanged;
   DeepR's static candidate CSR unchanged.
7. Rewire/add scoring using GradPreAct/Activation reads the batch-mean product;
   the shortlist/CollectProposalsKernel just calls the policy, so the only change
   is inside the policy via `BatchEdgeGradient`.
8. Input activations for NumInput units are batched too; H2D becomes a strided
   `B×NumInput` copy (ADR-007 zero-copy hands `[B, NumInput]` row-major -> pick
   unit-major on the plastix side + transpose on ingest, `B×785` floats).

## Memory & performance model

- **Memory:** batched unit columns = `n_batched × 4 × B × Cap`. DeepR has 5
  batched floats = 20 B/unit/sample. Cap=4096,B=64 -> 5.2 MB; Cap=1e6,B=64 ->
  1.28 GB; Cap=1e6,B=256 -> 5.1 GB. Batched state dominates the conn arena
  (~64 B/conn) only when `20·B·N_units > 64·N_conns`, i.e. `B > 3.2 × avg_degree`
  (~25-200 for degree 8-64). DeepR's dense candidate set makes conns dominate
  until very large B. Don't mirror batched columns on the host.
- **The win:** the ~100-250 µs/sample floor is paid once per B samples
  (~2-4 µs/sample at B=64) and occupancy rises from ~0.1% toward saturation with
  no extra launches. (Mirror of the plastax result: batched dense 584× faster than
  batch-1.) **Do the CUDA-Graph capture + drop per-phase `cudaDeviceSynchronize`
  (ADR-007 R3) FIRST** -- it changes the baseline batching is judged against
  ([CUDA-Graph kernel batching](https://arxiv.org/pdf/2501.09398)).
- **When batching stops helping:** once a single level's live edges exceed
  resident thread capacity (A100 ~2.2e5 threads; realistically ≥1e6 edges/level
  for memory-bound saturation), B=1 already fills the GPU; batching then only
  amortizes the fixed floor (a few %) while multiplying atomic traffic + activation
  bandwidth by B. Also does nothing for host phases (Sort, host add-conn, CPU
  rewire) -- they become the bottleneck if run every batch.
- **CPU backend:** gains only cache locality (contiguous `Act[src*B..]` row) +
  vectorization over b; no launch floor. Implement for parity/testing, not speed.

## Trait / API change surface (minimize user-policy churn)

- **`Traits::MaxBatch` (compile-time, default 1).** Sizes batched arenas + fixes
  the stride. `MaxBatch==1` must instantiate today's layout/kernels exactly
  (`Id*1+0`), so all existing examples/tests/parity vectors + online examples are
  untouched. Runtime `B ≤ MaxBatch` per call for the ragged last batch.
- **`SampleView<SoaView> = {view, b}`** whose `Get<Tag>(Id)` returns
  `Ptr[Id*MaxBatch+b]` for batched tags and `Ptr[Id]` for shared tags. Every
  existing `PLASTIX_HD` policy body (Map/Combine/Apply/CalculateLoss, DeepR
  Ltu*, MSELoss/RMSLoss/SoftmaxCE) compiles unchanged on a SampleView, because
  `GetField`/`GetActivation` forward to `Alloc.Get<Tag>(Id)`. Concepts checked
  against the sample-view type.
- **Batch-aware policies** get the full batched view + helpers:
  `BatchEdgeGradient<GradTag>(U,Dst,Src)`, `BatchMean<Tag>(U,Id)`,
  `BatchFraction(U,Id,pred)`, `G.BatchSize`. Only UpdateConn, PruneUnit,
  PruneConn, AddConn::Score bodies change, and only where they read a batched
  column: `optim.hpp` (one line), DeepR `DeepRConnUpdate` (line 101-102 ->
  `BatchEdgeGradient`), RigL Score. **Compile-time guard:** make
  `Get<BatchedTag>(Id)` ill-formed on the non-sample view when `MaxBatch>1` so a
  silent "uses sample 0" bug can't compile.
- **Entry points:** `DoForwardPass(span, B)` (`B*NumInput`, row-major, transposed
  on ingest) + the ADR-007 device-pointer twin; `DoCalculateLoss(Targets, B)`;
  `GetOutput() -> [B][Out]`; `DoStep(Inputs, Targets, B)`. `Step` counts batch
  steps; add `SamplesSeen`.
- Non-float Accumulator (SET MagnitudeStats) stays host-path; unchanged guard.

## Risks / unknowns

1. **MaxFieldSize padding** would ×B every column if batched fields are ordinary
   `SOAField`s -> needs the separate batched-column concept.
2. **Kernel param size:** `SoaView` is a tuple-of-pointers already flagged as not
   guaranteed trivially copyable; a second pointer pack + stride grows it ->
   convert to a pointer aggregate first.
3. **Determinism:** atomic order already nondeterministic per sample; batch-mean
   gradient is a fixed-order per-thread reduction (deterministic). Parity tests
   tolerance-based on the atomic path, exact only on RevAdj.
4. **Plastax parity:** plastax `StepInputs` is batch-1; B>1 parity needs plastax
   vmap+mean or restrict parity to B=1 -- decide in impl.
5. **Host phases at batch cadence** (Sort, host add-conn, CPU rewire, stale
   staging) dominate once GPU phases are batched.
6. **GlobalState accumulators** change meaning -> document + `G.BatchSize`.
7. **Pipeline model** batches only as independent streams.
8. **Register pressure** with register-blocked R samples/thread -> tune R, Bmax;
   `b`-innermost, R=1 is the safe default.
9. **Tail batches** waste `Bmax-B` lanes; acceptable.

## Phased plan (smallest win first)

- **P0 (measure + cheap win, no batching):** `nsys` the DeepR `DoStep` at B=1
  (quantify launches/syncs/bus crossings); add a stream parameter + drop the
  per-phase `cudaDeviceSynchronize` (ADR-007 R3); optionally CUDA-Graph-capture
  one `DoStep`. Gate: identical live-conn trace; measured µs/step before/after.
  Fixes the baseline.
- **P1 (batched forward-only inference):** batched-column allocator concept
  (`MaxBatch`, `[Cap][Bmax]`, device-only, outside MaxFieldSize), `SampleView`,
  batched forward sweep/apply/reduce kernels, `DoForwardPass(Inputs, B)`,
  `GetOutput [B][Out]`. Gate: `MaxBatch=1` bit-identical everywhere; B=64 outputs
  == 64 sequential forwards (tolerance on atomic path, exact on RevAdj); µs/sample
  vs B curve demonstrates the launch-amortization win.
- **P2 (batched loss + backward + gradient):** device loss kernel, batched
  backward, `BatchEdgeGradient`, switch `optim.hpp`, guard scalar reads of batched
  columns in batch-level phases. Gate: per-edge gradient == mean of B per-sample
  gradients; `optim` parity at B=1 unchanged.
- **P3 (batched dynamic training):** DeepR at the paper's B=10 and B=64
  end-to-end (update/prune/rewire once per batch); `G.BatchSize`; document
  per-algorithm semantics. Gate: live-conn trajectory statistically matches the
  notebook reference at matched B; throughput vs P0.
- **P4 (perf):** register blocking / float4 loads, sm_90 vector atomics, batched
  per-unit reduce tuning, CUDA-Graph capture of the whole batched step, fix the
  host-phase/stale-staging bottlenecks P3 exposes.

## Files most affected

`alloc.hpp`, `device_alloc.hpp` (batched column + sample view), `cuda_kernels.hpp`,
`dispatch_gpu.hpp`, `dispatch_cpu.hpp` (B-dimension kernels/loops), `plastix.hpp`
(`DoForwardPass`/`DoCalculateLoss`/`DoStep` shapes, `MaxBatch`), `traits.hpp`
(concepts on the sample view, `MaxBatch`, batch helpers), `optim.hpp`
(`EdgeGradient`); user side: `deepr_policies.hpp` (edge gradient + the loss slice
pushes the device loss kernel makes unnecessary).
