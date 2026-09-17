#ifndef PLASTIX_DEVICE_ALLOC_HPP
#define PLASTIX_DEVICE_ALLOC_HPP

// Opt-in device-resident SOA allocator (ADR-006). Unlike the default managed /
// mmap backend, this keeps the arena in an explicit cudaMalloc device buffer
// that stays device-resident for the duration of GPU execution, with a host
// staging mirror used only at construction (write + PushToDevice) and readback
// (PullToHost). No managed memory, so no per-step page-fault migration when a
// workload alternates host and device access (e.g. DEEP R's per-step loop).
//
// Host code writes/reads the staging view (Allocate/Get/Size forward to it);
// kernels take view(), which hands out the device buffer. The two are kept in
// sync explicitly via PushToDevice()/PullToHost() -- there is no coherence, by
// design. See notes/adr-006-device-resident-soa.md.

#include "plastix/alloc.hpp"

#ifdef PLASTIX_HAS_CUDA

#include "plastix/defer.hpp"
#include "plastix/macros.hpp"

#include <cstdlib>
#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace plastix {
namespace alloc {

// Device permutation-gather (device-resident compaction/resort, ADR-006
// follow-up): Dst[I] = Src[Perm[I]] for I in [0,N). One instantiation per field.
template <typename E>
__global__ void DeviceGatherKernel(E *Dst, const E *Src, const size_t *Perm,
                                   size_t N) {
  for (size_t I = blockIdx.x * blockDim.x + threadIdx.x; I < N;
       I += gridDim.x * blockDim.x)
    Dst[I] = Src[Perm[I]];
}

// Store a scalar live count into a device-resident atomic (post-compaction).
template <typename C>
__global__ void DeviceSetCountKernel(C *Count, size_t N) {
  Count->store(N);
}

// Device-resident twin of SOAAllocator: a host staging SoaView plus a device
// SoaView over cudaMalloc buffers. Move-only (one owner frees). Field arenas
// are over-allocated to Capacity*MaxFieldSize to match SOAAllocator's layout;
// the device back buffers / permutation scratch are left null because the
// device-resident path does no in-place compaction yet (ADR-006 follow-up).
template <typename T, typename... Fields> class DeviceResidentSOA {
  SoaView<T, Fields...> Host_;
  SoaView<T, Fields...> Device_;
  bool Owns_ = false;

  static constexpr size_t MaxFieldSize() {
    return std::max({sizeof(typename Fields::Type)...});
  }

  static void CudaTry(cudaError_t E, const char *What) {
    if (E != cudaSuccess)
      throw std::runtime_error(std::string("DeviceResidentSOA: ") + What +
                               ": " + cudaGetErrorString(E));
  }

  // Zeroed host storage (matches the mmap/managed zero-init contract that
  // Allocate and dead-slot reads rely on). Throws so the ctor's Defer unwinds.
  static void *HostAlloc(size_t Bytes) {
    void *P = std::calloc(1, Bytes);
    if (!P)
      throw std::bad_alloc();
    return P;
  }

  // Zeroed device storage; frees the buffer if the zeroing memset fails so the
  // helper never leaks on its own.
  static void *DeviceAlloc(size_t Bytes) {
    void *P = nullptr;
    CudaTry(cudaMalloc(&P, Bytes), "cudaMalloc");
    cudaError_t E = cudaMemset(P, 0, Bytes);
    if (E != cudaSuccess) {
      cudaFree(P);
      CudaTry(E, "cudaMemset");
    }
    return P;
  }

  void AllocHost(size_t N) {
    Host_.Capacity = N;
    const size_t Arena = N * MaxFieldSize();
    std::apply(
        [&](auto &...P) {
          ((P = static_cast<decltype(+P)>(HostAlloc(Arena))), ...);
        },
        Host_.FieldPtrs);
    std::apply(
        [&](auto &...P) {
          ((P = static_cast<decltype(+P)>(HostAlloc(Arena))), ...);
        },
        Host_.BackFieldPtrs);
    Host_.PermScratch = static_cast<size_t *>(HostAlloc(N * sizeof(size_t)));
    Host_.CountPtr = static_cast<AtomicCount *>(HostAlloc(sizeof(AtomicCount)));
    ::new (Host_.CountPtr) AtomicCount{0};
  }

  void AllocDevice(size_t N) {
    Device_.Capacity = N;
    const size_t Arena = N * MaxFieldSize();
    std::apply(
        [&](auto &...P) {
          ((P = static_cast<decltype(+P)>(DeviceAlloc(Arena))), ...);
        },
        Device_.FieldPtrs);
    // Device-resident growth (ADR-006 follow-up): back buffers + permutation
    // scratch back in-place device compaction/resort (GatherDevice).
    std::apply(
        [&](auto &...P) {
          ((P = static_cast<decltype(+P)>(DeviceAlloc(Arena))), ...);
        },
        Device_.BackFieldPtrs);
    Device_.PermScratch = static_cast<size_t *>(DeviceAlloc(N * sizeof(size_t)));
    Device_.CountPtr =
        static_cast<AtomicCount *>(DeviceAlloc(sizeof(AtomicCount)));
  }

  void ReleaseHost() {
    std::apply([](auto *...P) { ((std::free(P)), ...); }, Host_.FieldPtrs);
    std::apply([](auto *...P) { ((std::free(P)), ...); }, Host_.BackFieldPtrs);
    std::free(Host_.PermScratch);
    if (Host_.CountPtr) {
      Host_.CountPtr->~AtomicCount();
      std::free(Host_.CountPtr);
    }
    Host_ = SoaView<T, Fields...>{};
  }

  void ReleaseDevice() {
    // cudaFree(nullptr) is a well-defined no-op, so unset fields are safe.
    std::apply([](auto *...P) { ((void)cudaFree(P), ...); }, Device_.FieldPtrs);
    std::apply([](auto *...P) { ((void)cudaFree(P), ...); },
               Device_.BackFieldPtrs);
    cudaFree(Device_.PermScratch);
    cudaFree(Device_.CountPtr);
    Device_ = SoaView<T, Fields...>{};
  }

  template <size_t... Is>
  void GatherFieldsDevice(std::index_sequence<Is...>, size_t N, unsigned Grid,
                          unsigned Block) {
    ((DeviceGatherKernel<<<Grid, Block>>>(std::get<Is>(Device_.BackFieldPtrs),
                                          std::get<Is>(Device_.FieldPtrs),
                                          Device_.PermScratch, N)),
     ...);
  }

  template <size_t I> void CopyOneField(size_t N, cudaMemcpyKind Kind) {
    auto *H = std::get<I>(Host_.FieldPtrs);
    auto *D = std::get<I>(Device_.FieldPtrs);
    const size_t Bytes = N * sizeof(*H);
    if (Kind == cudaMemcpyHostToDevice)
      CudaTry(cudaMemcpy(D, H, Bytes, Kind), "H2D field");
    else
      CudaTry(cudaMemcpy(H, D, Bytes, Kind), "D2H field");
  }

  template <size_t... Is>
  void CopyFields(std::index_sequence<Is...>, size_t N, cudaMemcpyKind Kind) {
    ((CopyOneField<Is>(N, Kind)), ...);
  }

public:
  explicit DeviceResidentSOA(size_t NumElements) {
    // Free anything already allocated if a later allocation throws; dismissed
    // once construction fully succeeds (RAII owns teardown from then on).
    Defer Guard([this] {
      ReleaseHost();
      ReleaseDevice();
    });
    AllocHost(NumElements);
    AllocDevice(NumElements);
    Owns_ = true;
    Guard.dismiss();
  }

  ~DeviceResidentSOA() {
    if (Owns_) {
      ReleaseHost();
      ReleaseDevice();
    }
  }

  DeviceResidentSOA(const DeviceResidentSOA &) = delete;
  DeviceResidentSOA &operator=(const DeviceResidentSOA &) = delete;

  DeviceResidentSOA(DeviceResidentSOA &&Other) noexcept
      : Host_{Other.Host_}, Device_{Other.Device_}, Owns_{Other.Owns_} {
    Other.Owns_ = false;
    Other.Host_ = SoaView<T, Fields...>{};
    Other.Device_ = SoaView<T, Fields...>{};
  }
  DeviceResidentSOA &operator=(DeviceResidentSOA &&Other) noexcept {
    if (this != &Other) {
      if (Owns_) {
        ReleaseHost();
        ReleaseDevice();
      }
      Host_ = Other.Host_;
      Device_ = Other.Device_;
      Owns_ = Other.Owns_;
      Other.Owns_ = false;
      Other.Host_ = SoaView<T, Fields...>{};
      Other.Device_ = SoaView<T, Fields...>{};
    }
    return *this;
  }

  // Host-side construction API (writes the staging mirror).
  AllocId<T> Allocate() { return Host_.Allocate(); }
  std::pair<size_t, size_t> AllocateMany(size_t N) {
    return Host_.AllocateMany(N);
  }

  template <typename FieldTag> auto &Get(AllocId<T> Id) {
    return Host_.template Get<FieldTag>(Id);
  }
  template <typename FieldTag> const auto &Get(AllocId<T> Id) const {
    return Host_.template Get<FieldTag>(Id);
  }
  auto &Get(AllocId<T> Id)
    requires(sizeof...(Fields) == 1)
  {
    return Host_.Get(Id);
  }
  const auto &Get(AllocId<T> Id) const
    requires(sizeof...(Fields) == 1)
  {
    return Host_.Get(Id);
  }
  template <typename FieldTag> auto *GetArrayFor() {
    return Host_.template GetArrayFor<FieldTag>();
  }
  template <typename FieldTag> const auto *GetArrayFor() const {
    return Host_.template GetArrayFor<FieldTag>();
  }

  size_t Size() const { return Host_.Size(); }
  size_t GetCapacity() const { return Host_.GetCapacity(); }
  size_t *PermutationScratch() { return Host_.PermutationScratch(); }

  // Host-staging reorder (construction-time resort). Operates on the staging
  // mirror; a following PushToDevice makes the reordered data authoritative.
  void Gather(size_t N) { Host_.Gather(N); }
  // Set the live count on the staging mirror; PushToDevice syncs it to device.
  void SetCount(size_t N) { Host_.SetCount(N); }

  // Device handle for kernel launches (the cudaMalloc buffers).
  SoaView<T, Fields...> view() const { return Device_; }

  // ---- Device-resident growth (ADR-006 follow-up) --------------------------
  // Permutation scratch on the device buffer; a kernel fills it, GatherDevice
  // consumes it. No host round-trip.
  size_t *DevicePermScratch() { return Device_.PermScratch; }

  // Device-side reorder: scatter Field[Perm[I]] -> Back[I] for I in [0,N), then
  // swap primary<->back so the reordered arrays are authoritative on device.
  // Fill DevicePermScratch() first. Stays entirely on the device buffer.
  void GatherDevice(size_t N) {
    if (N > 0) {
      const unsigned Block = 256;
      const unsigned Grid = static_cast<unsigned>((N + Block - 1) / Block);
      GatherFieldsDevice(std::index_sequence_for<Fields...>{}, N, Grid, Block);
      CudaTry(cudaGetLastError(), "GatherDevice launch");
      CudaTry(cudaDeviceSynchronize(), "GatherDevice sync");
    }
    std::swap(Device_.FieldPtrs, Device_.BackFieldPtrs);
  }

  // Set the device-resident live count (post-compaction extent), no host copy.
  void SetCountDevice(size_t N) {
    DeviceSetCountKernel<<<1, 1>>>(Device_.CountPtr, N);
    CudaTry(cudaGetLastError(), "SetCountDevice launch");
    CudaTry(cudaDeviceSynchronize(), "SetCountDevice sync");
  }

  // Read the device-resident live count to host (a kernel may have grown it).
  size_t DeviceCount() const {
    size_t N = 0;
    CudaTry(cudaMemcpy(&N, Device_.CountPtr, sizeof(size_t),
                       cudaMemcpyDeviceToHost),
            "D2H device count");
    return N;
  }

  // Targeted one-field slice sync: copy [Begin, Begin+Count) of field Tag across
  // the boundary without touching the rest of the arena. Used for the small
  // per-step host<->device exchanges (input activations in, output slice out)
  // so the bulk of the arena stays device-resident.
  template <typename Tag> void PushFieldSlice(size_t Begin, size_t Count) {
    auto *H = Host_.template GetArrayFor<Tag>();
    auto *D = Device_.template GetArrayFor<Tag>();
    CudaTry(cudaMemcpy(D + Begin, H + Begin, Count * sizeof(*H),
                       cudaMemcpyHostToDevice),
            "H2D field slice");
  }
  template <typename Tag> void PullFieldSlice(size_t Begin, size_t Count) {
    auto *H = Host_.template GetArrayFor<Tag>();
    auto *D = Device_.template GetArrayFor<Tag>();
    CudaTry(cudaMemcpy(H + Begin, D + Begin, Count * sizeof(*H),
                       cudaMemcpyDeviceToHost),
            "D2H field slice");
  }

  // Explicit sync across the host/device boundary (no coherence by design).
  void PushToDevice() {
    const size_t N = Host_.Size();
    CopyFields(std::index_sequence_for<Fields...>{}, N,
               cudaMemcpyHostToDevice);
    CudaTry(cudaMemcpy(Device_.CountPtr, Host_.CountPtr, sizeof(AtomicCount),
                       cudaMemcpyHostToDevice),
            "H2D count");
  }

  void PullToHost() {
    // Count first: a kernel may have grown it, and it sets the field extent.
    CudaTry(cudaMemcpy(Host_.CountPtr, Device_.CountPtr, sizeof(AtomicCount),
                       cudaMemcpyDeviceToHost),
            "D2H count");
    const size_t N = Host_.Size();
    CopyFields(std::index_sequence_for<Fields...>{}, N,
               cudaMemcpyDeviceToHost);
  }
};

template <typename T, typename... Fields>
inline constexpr bool IsDeviceResidentV<DeviceResidentSOA<T, Fields...>> = true;

} // namespace alloc
} // namespace plastix

#endif // PLASTIX_HAS_CUDA

#endif // PLASTIX_DEVICE_ALLOC_HPP
