# ADR-007: Python bindings for user-defined plastix networks

Status: proposed (research/design; nothing implemented)
Related: ADR-006 (device-resident SOA)
Research: Fable-class agent, 2026-09-16. Sources cited inline.

## Context

Goal: let a user define their network in C++ (as today) and drive it from a
Python script with a PyTorch `DataLoader` feeding batches (CPU or CUDA tensors),
to train/eval on existing datasets. The device-resident GPU backend (ADR-006)
makes a zero-copy GPU data path possible: a CUDA tensor could feed a
device-resident SOA field with no host round-trip.

## Facts about plastix that constrain the design

1. **A network is a compile-time type.** `template <NetworkTraits Traits> class
   Network` (`plastix.hpp:68`); allocators `UnitAllocFor<T>`/`ConnAllocFor<T>`
   and every `Do*` phase are selected by `if constexpr` on the traits
   (`traits.hpp:323-334`, `plastix.hpp:173-385`). The ctor takes user builder
   *functors* (`SparseFanIn{...}`, `OneToOneRouting{...}`) + an input-init
   functor. There is no runtime description of a network; two traits structs are
   two unrelated C++ types.
2. **Header-only, and the GPU path must be compiled by nvcc.**
   `plastix_enable_cuda_on_target` (`CMakeLists.txt:59-81`) flips every `.cpp` of
   a target to `LANGUAGE CUDA` because `PLASTIX_HD` code and the `__global__`
   templates in `dispatch_gpu.hpp`/`cuda_kernels.hpp` are instantiated in the
   consumer's TU. Any TU that instantiates `Network<T>` under CUDA is an nvcc TU
   (`--extended-lambda --expt-relaxed-constexpr`, separable compilation, arches
   `80;86;89;90`).
3. **Data path today is host-staged.** `DoForwardPass(std::span<const float>)`
   writes input activations into the host staging mirror then
   `PushFieldSlice<ActivationTag>(0, NumInput)` (H2D, `plastix.hpp:177-182`,
   `device_alloc.hpp:232-238`). `GetOutput()` returns a host span; the
   device-resident path `SyncOutputToHost()` D2H-pulls the output slice first
   (`plastix.hpp:387-400`). `DoCalculateLoss` runs the loss policy on the host
   view (`plastix.hpp:215`) -- on the device-resident backend that is the staging
   mirror, so it implicitly needs the output slice pulled.
4. **All kernels launch on the legacy default stream**; every phase ends with
   `cudaDeviceSynchronize()` (`dispatch_gpu.hpp:45-82`). No stream parameter.
5. **`Network` is non-copyable/non-movable** (`plastix.hpp:166-169`);
   `DeepRNetwork<Traits>` is a composition wrapper with its own `DoStep`
   (`plastix_deepr.hpp`). Runtime hyperparameters live in `Global()` (a plain
   aggregate).
6. Output range / NumInput are private; `GetOutput()` gives only a host span. The
   binding needs `OutputRange.Begin` to build a device view.

## Decision

### Primary: nanobind + a reusable non-template facade + a ~5-line user shim, built with scikit-build-core under uv

```
plastix/include/plastix/py/
  handle.hpp    # NetworkHandle: abstract, NON-template interface (no nanobind include)
  adapter.hpp   # template <class Net> Adapter : NetworkHandle  (instantiated in the user's nvcc TU)
  module.hpp    # nanobind binding of NetworkHandle + PLASTIX_PY_MODULE(...) macro
user-project/
  src/mynet_py.cpp   # ~5 lines: include traits, call PLASTIX_PY_MODULE
  CMakeLists.txt, pyproject.toml
  train.py           # torch DataLoader loop
```

Why this shape:

- **Type erasure at the C++ level, once.** `NetworkHandle` is a small virtual
  interface. `Adapter<Net>` implements it over *any* class with plastix's
  `Do*`/`GetOutput`/`GetUnitAlloc` surface (both `Network<T>` and wrappers like
  `DeepRNetwork<T>` qualify -- checked by a concept). The user's TU instantiates
  exactly one `Adapter<TheirNet>`; that is the only place user policies meet the
  compiler, so it is the only nvcc TU.
- **nanobind sees only the facade.** The macro binds `NetworkHandle`
  generically; the same binding code is reused verbatim for every user network,
  and (via the two-TU split, R1) nanobind's headers need not go through nvcc.
- **nanobind over pybind11:** `nb::ndarray` is the only mainstream caster that
  natively ingests **DLPack (versioned since 2.10.1) and the buffer protocol**
  from NumPy/PyTorch/JAX/CuPy, with `nb::device::cuda` constraints and
  `device_id()` for multi-GPU, and can *export* a `nb::pytorch`-tagged array over
  a raw device pointer ([ndarray docs](https://nanobind.readthedocs.io/en/latest/ndarray.html)).
  pybind11 has only NumPy `array_t`; DLPack/`__cuda_array_interface__` remain
  open issues ([#3858](https://github.com/pybind/pybind11/issues/3858),
  [#2694](https://github.com/pybind/pybind11/issues/2694)). nanobind compiles
  2.7-4.4x faster / 3-5x smaller binaries
  ([benchmarks](https://nanobind.readthedocs.io/en/latest/benchmark.html));
  latest 3.0.1 (Aug 2026). 2026 consensus for new bindings is nanobind
  ([matecdev](https://www.matecdev.com/posts/nanobind-vs-pybind11-cpp-python.html)).

### Fallback: same facade exported as `extern "C"`, consumed by ctypes/cffi

`handle.hpp` is nanobind-free, so a ~40-line `plastix_c.h`
(`plastix_net_create(cfg)`, `plastix_net_step(h, x, y)`,
`plastix_net_forward_device(h, d_x)`, `plastix_net_output_device(h)`) gives a
binding with zero third-party dependency, immune to nvcc-vs-binding-header
issues, callable from anything (ctypes, cffi, Julia, Mojo via `external_call`).
Data goes in as `tensor.data_ptr()`. Loses ergonomics (no `__init__(**kwargs)`,
manual dtype/shape checks, manual DLPack). Pick this if R1 bites. pybind11 3.x
(July 2025) is a near-drop-in *alternative* for the facade binding but does not
solve the GPU data path, so it is not the fallback.

## The central challenge: user-defined compile-time networks

| Option | Verdict | Why |
|---|---|---|
| **1. Per-network shim + reusable header/macro** | **Recommended** | The user already writes C++ (traits, policies, builders, `main.cpp`). Replacing `main.cpp` with a 5-line `mynet_py.cpp` is strictly less work. The shim is the only place the compiler sees their types, so it is the only nvcc TU. Build cost = same as their `main_deepr` executable today (+ nanobind core). |
| **2. cppyy/cling runtime instantiation** | **Not viable** | cppyy 3.5.0 runs on Clang 16, C++20 since 3.1, but packaging is uneven (conda-forge builds reject `concept`, [#323](https://github.com/wlav/cppyy/issues/323)). CUDA mode is "experimental… incomplete… only tested on Linux x86_64," pinned to old Clang/CUDA (docs: Clang13 / "CUDA up to 10.2"), "no syntax or helpers yet to launch kernels from Python" ([cppyy CUDA docs](https://cppyy.readthedocs.io/en/latest/cuda.html)). Our device-resident backend needs CUDA 12.6+/13, sm_80+, `--extended-lambda`, separable compilation -- none provided. Per-process JIT of every phase is slow (seconds-tens uncached). CPU-only notebook prototyping at best. |
| **3. Codegen from traits** | **Not worth it now** | C++20 has no reflection; a libclang generator would emit the same 5 lines the macro produces. Cheaper enumeration of `ExtraUnit/ConnFields`/`GlobalState`: a `PLASTIX_PY_GLOBALS(...)` FOR_EACH macro, or Boost.PFR `for_each_field`/`names_as_array` (C++20) for aggregate `GlobalState`; C++26 reflection later. |

## Binding-technology tradeoff table

(header-only C++20 templates + CUDA + user instantiations)

| Tech | C++20 templates/concepts | CUDA device code in binding TU | GPU zero-copy in | GPU zero-copy out | uv + scikit-build-core | Stable ABI / free-threading | Fit |
|---|---|---|---|---|---|---|---|
| **nanobind 3.0.1** | Yes (>=C++17) | Yes, nvcc supported + CI since 2.2.0 | **Yes** `nb::ndarray<const float, nb::device::cuda>` via DLPack >=2.10.1 | **Yes** `nb::ndarray<nb::pytorch>` over device ptr | First-class (`nanobind_add_module`, `uv init --build-backend scikit-build-core`) | Yes (3.12+, or 3.10 split) / yes since 2.2 | **Best** |
| pybind11 3.0.x | Yes | Yes (static cudart) | Plugin/hand-rolled DLPack only | Hand-rolled | Good | Partial / `mod_gil_not_used` | OK; larger, slower, no ndarray |
| cppyy 3.5 | C++20 on Clang16; packaging gaps | **No** (experimental, CUDA <=10.2-era) | No | No | ships own LLVM | n/a | CPU notebook only |
| Cython 3.x | templates/concepts awkward; policies undeclarable | via nvcc `.cu` + C-ABI only | Manual | Manual | Works | Limited-API partial | Poor |
| cffi / ctypes (C-ABI) | Irrelevant (C ABI) | Yes -- shim is nvcc TU, ctypes never sees it | Manual `data_ptr()` + stream | Manual (build CAI dict) | Trivial | Perfect | **Fallback** |
| SWIG 4.x | C++20 minimal; concepts unparsed | via generated C++ | No | No | Weak | n/a | Not viable |
| **Mojo 1.0 (Aug 2026)** | **No C++ interop** (only `extern "C"`/`external_call`/`DLHandle`; [FAQ](https://docs.modular.com/mojo/faq), [roadmap](https://docs.modular.com/mojo/roadmap)) | No (calls our C-ABI shim) | community torch DLPack only | same | `PythonModuleBuilder` (max 6 args, no properties) | n/a | **Not yet** |

## Prototype design

### `include/plastix/py/handle.hpp` -- nanobind-free facade (core, one-time)

```cpp
namespace plastix::py {
struct FieldDesc { const char* name; char dtype; size_t itemsize; size_t count;
                   const void* host; const void* device; };   // dtype: 'f','u','i','b'
struct NetworkHandle {
  virtual ~NetworkHandle() = default;
  virtual size_t NumInput() const = 0;
  virtual size_t NumOutput() const = 0;
  virtual bool   DeviceResident() const = 0;
  virtual void   Forward(const float* x) = 0;                   // host x
  virtual void   Step(const float* x, const float* y) = 0;      // host x,y (y may be null)
  virtual void   StepBatch(const float* X, const float* Y, size_t B) = 0; // rows, loop in C++
  virtual void   ForwardDevice(const float* d_x, void* stream) = 0;       // NEW core hook
  virtual void   StepDevice(const float* d_x, const float* d_y, void* stream) = 0;
  virtual const float* OutputHost() = 0;          // SyncOutputToHost() then GetOutput().data()
  virtual const float* OutputDevice() const = 0;  // device activation ptr + OutputRange.Begin
  virtual std::span<const FieldDesc> Fields() const = 0;
  virtual size_t StepCount() const = 0;
};
}
```

### `include/plastix/py/adapter.hpp` -- `template <class Net> Adapter : NetworkHandle`

Implements the above over any `Net` satisfying `requires { n.DoForwardPass(span);
n.DoStep(span, span); n.GetOutput(); n.GetUnitAlloc(); n.GetConnAlloc(); }`.
`Fields()` is filled from a user tag list (`plastix::py::Field<deepr::ThetaTag>("theta")`),
each capturing `GetArrayFor<Tag>()` on the host view and, when
`IsDeviceResidentV`, on `.view()` (device). Requires two tiny core additions: a
public `GetOutputRange()`/`GetNumInput()` on `Network`, and the device-input
entry point. `DeepRNetwork` gets the same two forwarding one-liners.

### `include/plastix/py/module.hpp` -- generic nanobind binding + macro

```cpp
#define PLASTIX_PY_MODULE(ModName, Factory, ...)                                  \
  NB_MODULE(ModName, m) {                                                         \
    plastix::py::BindNetwork(m, /*factory*/ Factory, /*globals*/ __VA_ARGS__);    \
  }
```

`BindNetwork` defines `nb::class_<NetworkHandle>(m, "Network")` with:
- `nb::new_([](nb::kwargs kw){ return Factory(Args{kw}); })` ->
  `std::unique_ptr<NetworkHandle>` (`nb::new_` since `Network` is non-movable and
  the base is abstract);
- `forward(x)`, `step(x, y=None)`, `step_batch(X, Y)` taking
  `nb::ndarray<const float, nb::ndim<1|2>, nb::c_contig, nb::device::cpu>`
  (buffer protocol/NumPy + CPU torch, zero-copy);
- `forward_device(x, stream)`, `step_device(x, y, stream)` taking
  `nb::ndarray<const float, nb::c_contig, nb::device::cuda>`;
- `output` property -> `nb::ndarray<nb::numpy, const float>` (host) and
  `output_device` -> `nb::ndarray<nb::pytorch, float>(ptr, {n}, owner=self, {},
  nb::dtype<float>(), nb::device::cuda::value, dev)` (zero-copy torch view);
- `field(name)` -> ndarray over the SOA column (host or device flavour);
- `globals` object with `def_rw` per name from `PLASTIX_PY_GLOBALS(...)`;
- all compute methods wrapped in `nb::call_guard<nb::gil_scoped_release>()` so
  DataLoader workers keep prefetching.

### User shim -- `src/deepr_py.cpp` (compiled as CUDA by `plastix_enable_cuda_on_target`)

```cpp
#include "plastix_deepr.hpp"           // their traits/policies/layers, unchanged
#include <plastix/py/module.hpp>
using Traits = deepr::DeepRTraitsT<4096, (1u << 15)>;
PLASTIX_PY_MODULE(deepr_py,
  [](plastix::py::Args a) { return plastix::py::Wrap<deepr::DeepRNetwork<Traits>>(
        a.get<size_t>("inputs"), deepr::DeepRInputInit{},
        deepr::SparseFanIn{a.get<size_t>("hidden"), a.get<uint32_t>("fan_in"), 0, 1u},
        deepr::OneToOneRouting{a.get<size_t>("outputs"), 1, 2u, 10},
        plastix::py::Field<deepr::ThetaTag>("theta"), plastix::py::Field<plastix::DeadTag>("dead")); },
  PLASTIX_PY_GLOBALS(deepr::DeepRGlobals, Seed, InitMagnitude, MaxGenPerStep, AgeGateEnabled))
```

That is the entire replacement for `main_deepr.cpp`.

### Build config

`pyproject.toml` (patterns from [nanobind packaging](https://nanobind.readthedocs.io/en/latest/packaging.html),
[scikit-build-core config](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html),
[pybind/cuda_example](https://github.com/pybind/cuda_example)):

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.10"]
build-backend = "scikit_build_core.build"

[project]
name = "deepr-py"; version = "0.1.0"; requires-python = ">=3.10"
dependencies = ["numpy", "torch"]

[tool.scikit-build]
build-dir = "build/{wheel_tag}"
cmake.build-type = "Release"
editable.rebuild = true                     # rebuild-on-import for dev loops
[tool.scikit-build.cmake.define]
PLASTIX_ENABLE_CUDA = {env = "PLASTIX_ENABLE_CUDA", default = "OFF"}
CMAKE_CUDA_ARCHITECTURES = {env = "CMAKE_CUDA_ARCHITECTURES", default = "native"}
PLASTIX_DIR = {env = "PLASTIX_DIR", default = ""}   # for find_package(plastix)

[tool.uv]
cache-keys = [{file = "pyproject.toml"}, {file = "CMakeLists.txt"}, {file = "src/**"}]
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.26)
project(deepr_py LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(plastix CONFIG REQUIRED)        # exports plastix::plastix, PLASTIX_HAS_CUDA + CCCL
nanobind_add_module(_deepr_py NB_STATIC src/deepr_py.cpp)
target_link_libraries(_deepr_py PRIVATE plastix::plastix)
target_include_directories(_deepr_py PRIVATE ${CMAKE_SOURCE_DIR}/../plastix)  # deepr headers
if(PLASTIX_ENABLE_CUDA)
  enable_language(CUDA); set(CMAKE_CUDA_STANDARD 20)
  plastix_enable_cuda_on_target(_deepr_py)   # flips deepr_py.cpp to CUDA (export this helper
                                             # from plastixConfig.cmake -- small core change)
  set_target_properties(_deepr_py PROPERTIES CUDA_RUNTIME_LIBRARY Static)  # avoid clashing
                                             # with torch's bundled libcudart
endif()
install(TARGETS _deepr_py LIBRARY DESTINATION deepr_py)
```

Workflow: `uv init --lib --build-backend scikit-build-core` scaffolds this;
`PLASTIX_ENABLE_CUDA=ON uv sync` builds;
`uv pip install -e . -Ccmake.define.PLASTIX_ENABLE_CUDA=ON` overrides per install;
`uv sync --reinstall` forces a rebuild. Editable dev loop follows the nanobind
recipe `pip install --no-build-isolation -Ceditable.rebuild=true -ve .` (build
deps pre-installed; see [pytorch-nanobind-cuda-example](https://github.com/jannismoeller/pytorch-nanobind-cuda-example)).

### Python usage with a torch `DataLoader`

```python
import torch, torch.nn.functional as F
from torchvision import datasets, transforms
from deepr_py import Network

dev = torch.device("cuda")
net = Network(inputs=785, hidden=1024, fan_in=64, outputs=20)   # kwargs -> Factory
net.globals.Seed = 7; net.globals.MaxGenPerStep = 8

ds = datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

stream = torch.cuda.current_stream()
for imgs, labels in dl:
    x = torch.cat([imgs.flatten(1), torch.ones(imgs.size(0), 1)], 1).to(dev, non_blocking=True)
    y = F.one_hot(labels, 10).float().to(dev, non_blocking=True)
    # plastix is online/per-sample: step_device loops the batch rows in C++ with the GIL released
    net.step_device(x, y, stream=stream.cuda_stream)              # no host round-trip
    out = net.output_device                                        # zero-copy torch CUDA view

theta = torch.from_dlpack(net.field("theta"))       # zero-copy view of the SOA column (device)
alive = net.field("dead", device="cpu")             # host-pulled copy for metrics
```

CPU-only build: identical with `imgs.flatten(1).numpy()` / `net.step_batch(X, Y)`.

## GPU zero-copy data path (DLPack/CAI -> device-resident SOA)

**Ingress (torch CUDA tensor -> input activation slice).**
1. nanobind imports via `tensor.__dlpack__(max_version=…)`, preferring
   `__dlpack__` over the buffer protocol; `nb::device::cuda` rejects CPU tensors,
   `nb::c_contig` rejects strided views (`.contiguous()` in Python). Result: raw
   `const float*` device ptr + `device_id()`, no copy.
2. **New core method:** `Network::DoForwardPassDevice(const float* d_x,
   cudaStream_t s)` -> `cudaMemcpyAsync(Device_.GetArrayFor<ActivationTag>(),
   d_x, NumInput*4, D2D, s)` then the existing kernels, skipping the host staging
   write + `PushFieldSlice`. True aliasing is impossible (input activations sit at
   a fixed offset in the contiguous per-field arena), but a D2D copy of ~785
   floats is ~us and stays on the GPU -- the ADR-006 synergy: input never touches
   the host. Targets: `DoCalculateLossDevice(const float* d_y)` needs a
   **device-side loss kernel** (today `CalculateLoss` runs on the host view,
   forcing a D2H of the output slice every step). Loss policies are `PLASTIX_HD`
   already, so a one-thread-per-output wrapper is straightforward.
3. **Stream ordering (subtle).** nanobind passes no `stream` to `__dlpack__` and
   does no sync; torch's `Tensor.__dlpack__` defaults to `stream=-1` ("no
   synchronization"). Two safe options: (a) rely on legacy-default-stream
   semantics (plastix launches on stream 0, which syncs with blocking streams --
   torch's default/`Stream()` are blocking -- unless torch uses per-thread default
   stream); (b) explicit: Python passes `torch.cuda.current_stream().cuda_stream`,
   the binding does `cudaEventRecord(e, torch_stream);
   cudaStreamWaitEvent(plastix_stream, e)` before the D2D. **Recommend (b)** in
   `ForwardDevice(d_x, stream)`; costs nothing, robust. Egress is covered by
   plastix's per-phase `cudaDeviceSynchronize()`.
4. Multi-GPU: check `a.device_id()` against plastix's device; raise otherwise.

**Egress (device SOA slice -> torch).** `nb::ndarray<nb::pytorch, float>(dev_ptr
+ OutputRange.Begin, {NumOutput}, nb::find(self), {}, nb::dtype<float>(),
nb::device::cuda::value, dev_id)` returns a `torch.Tensor` aliasing the arena;
`owner=self` keeps the network alive. Same constructor exports any SOA column, so
`torch.from_dlpack(net.field("theta"))` is a live view. Caveat: after
`Gather`/compaction a column is permuted; treat views as valid until the next
`step`.

**`__cuda_array_interface__`:** not needed on ingress (DLPack subsumes it); the
C-ABI fallback would use it on egress (`torch.as_tensor(obj_with_cai)`).

**Batching gap:** `DoForwardPass` is one sample; plastix is online (per-sample
update, dynamic topology), so "batch" today = loop rows in C++ (`StepBatch`),
which removes Python per-sample overhead but not per-sample kernel launches (~10
launches + `cudaDeviceSynchronize` per phase per sample). Real batched forward
needs a batch dimension in the activation SOA and batched loss/backward -- a core
design change (see ADR-008). It is the main throughput limiter for MNIST-scale
training from Python. Removing the per-phase `cudaDeviceSynchronize` in favour of
a stream parameter is the cheaper first win.

## Small core additions required (all additive)

- `GetOutputRange()` / `GetNumInput()` accessors on `Network`.
- `DoForwardPassDevice(const float*, cudaStream_t)` + a device-side loss kernel.
- Export `plastix_enable_cuda_on_target` from `plastixConfig.cmake` (today it
  lives only in the top-level `CMakeLists.txt`, so `find_package(plastix)`
  consumers cannot call it).

## Risks / unknowns

- **R1 nanobind headers through nvcc.** Supported + CI-tested (2.2.0) and CMake
  gates flags, but our TU also carries `--extended-lambda`, C++20, separable
  compilation, heavy `if constexpr`. Mitigation: the facade split lets the
  nanobind TU be plain C++ with only `adapter.hpp` through nvcc (two-TU variant);
  keep both, pick after M1.
- **R2 CUDA runtime coexistence with torch.** torch wheels bundle their own
  cudart. Link the extension's cudart statically (`CUDA_RUNTIME_LIBRARY Static`);
  device pointers are context-level so sharing works; different *major* toolkits
  in one process is the untested corner.
- **R3 Legacy default stream + `cudaDeviceSynchronize` everywhere.** Correct but
  serialising; blocks overlap with DataLoader H2D. Needs the stream-parameter
  refactor (small, mechanical).
- **R4 Missing core hooks.** See "Small core additions".
- **R5 Field/global enumeration is manual** (no reflection). FOR_EACH macro or
  Boost.PFR now; C++26 reflection later.
- **R6 cppyy/C++20-in-JIT** -- not on the path; CPU-only, Clang-16-level.
- **R7 Mojo** -- no C++ interop in 1.0.
- **R8 Batching** -- see ADR-008; the biggest gap between "works" and "trains
  MNIST at useful speed".
- **R9 Compile time per network.** Each network recompiles plastix for every
  arch; default `native` in dev, full list only for wheels.

## Sequenced prototype

- **M0 (CPU, 1-2 days):** `handle.hpp` + `adapter.hpp` + `module.hpp` with
  `Network(**kwargs)`, `step`, `step_batch`, `output`, `globals`, `field`; DeepR
  shim; `uv sync` build; reproduce `main_deepr.cpp`'s live-conn trace bit-for-bit
  from Python on the synthetic stream. Gate: identical `live conns` at steps
  0/50/100/150/199.
- **M1 (CUDA, host path):** same shim `PLASTIX_ENABLE_CUDA=ON`,
  `CMAKE_CUDA_ARCHITECTURES=native`; nvcc through `nanobind_add_module`; static
  cudart; decide one-TU vs two-TU (R1). Gate: matches the CPU trace; torch
  imported in the same process without cudart conflicts.
- **M2 (GPU zero-copy):** add `DoForwardPassDevice` + device loss + stream arg;
  `step_device`/`output_device`/`field(...)` as torch views; MNIST `DataLoader`
  loop; measure per-sample time vs `main_deepr` and confirm with `nsys` that no
  per-step H2D/D2H occurs beyond requested metrics.
