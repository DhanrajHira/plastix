#ifndef PLASTIX_ALLOC_HPP
#define PLASTIX_ALLOC_HPP

#include "plastix/macros.hpp"

#include <algorithm>
#include <cstddef>
#include <new>
#include <tuple>
#include <utility>

#ifdef PLASTIX_HAS_CUDA
#include <cstdio>
#include <cstdlib>
#include <cuda/std/atomic>
#include <cuda_runtime.h>
#define PLASTIX_CUDA_CHECK(stmt)                                               \
  do {                                                                         \
    cudaError_t Err__ = (stmt);                                                \
    if (Err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(Err__));                                 \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#else
#include <atomic>
#include <sys/mman.h>
#endif

namespace plastix {
namespace alloc {

template <typename T> using AllocId = size_t;

// Atomic backing for `Count`. Under CUDA we need libcu++ so the same atomic
// can be touched from both host and device kernels. Outside CUDA the standard
// std::atomic is fine.
#ifdef PLASTIX_HAS_CUDA
using AtomicCount = cuda::std::atomic<size_t>;
#else
using AtomicCount = std::atomic<size_t>;
#endif

template <typename FieldTag, typename T>

struct SOAField {
  static_assert(std::is_class_v<FieldTag> && std::is_empty_v<FieldTag>,
                "Type FieldTag here is supposed to be an empty marker type. "
                "All the fields that "
                "this type should store should be passed in separately.");
  using Tag = FieldTag;
  using Type = T;
};

// Storage backend abstraction. Under PLASTIX_HAS_CUDA we use
// cudaMallocManaged so the same pointer is valid from both host and device.
// This is a deliberate deviation from the explicit-copy plan in
// notes/gpu-architecture.md: managed memory is significantly simpler to
// reason about, keeps the SOA API contract identical between backends, and
// is a reasonable starting point for an MVP. Tightening to explicit copies
// can be done later without churning user-facing code.
// TODO(plastix-gpu): switch to explicit cudaMalloc + staging copies.
namespace detail {

inline void *AllocStorage(size_t Bytes) {
#ifdef PLASTIX_HAS_CUDA
  void *P = nullptr;
  PLASTIX_CUDA_CHECK(cudaMallocManaged(&P, Bytes));
  PLASTIX_CUDA_CHECK(cudaMemset(P, 0, Bytes));
  return P;
#else
  return mmap(nullptr, Bytes, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#endif
}

inline void FreeStorage(void *P, size_t Bytes) {
#ifdef PLASTIX_HAS_CUDA
  (void)Bytes;
  if (P)
    PLASTIX_CUDA_CHECK(cudaFree(P));
#else
  if (P)
    munmap(P, Bytes);
#endif
}

} // namespace detail

template <typename T, typename... Fields> class SOAAllocator {
  std::tuple<typename Fields::Type *...> FieldPtrs;
  std::tuple<typename Fields::Type *...> BackFieldPtrs;
  size_t *PermScratch;
  // Heap allocation so the atomic lives in managed memory (for CUDA) and the
  // allocator object itself stays trivially movable across host/device.
  AtomicCount *CountPtr;
  size_t Capacity;
  // True for the unique owning instance constructed via `SOAAllocator(N)`.
  // Shallow copies (used to pass the allocator by value into CUDA kernels)
  // share the same backing pointers but must not free them on destruction.
  bool Owns;

  template <typename FieldTag> constexpr static size_t IndexOf() {
    size_t I = 0;
    ((std::is_same_v<FieldTag, typename Fields::Tag> ? false : (++I, true)) &&
     ...);
    return I;
  }

  static constexpr size_t MaxFieldSize() {
    return std::max({sizeof(typename Fields::Type)...});
  }

public:
  // Mark host/device callable accessors. The hot loop inside CUDA kernels
  // calls Get<Tag>() per element, so it must be device-callable.
  explicit SOAAllocator(size_t NumElements)
      : FieldPtrs{}, BackFieldPtrs{}, PermScratch{nullptr}, CountPtr{nullptr},
        Capacity{NumElements}, Owns{true} {
    size_t ArenaSize = NumElements * MaxFieldSize();
    std::apply(
        [ArenaSize](auto &...Ptrs) {
          ((Ptrs =
                static_cast<decltype(+Ptrs)>(detail::AllocStorage(ArenaSize))),
           ...);
        },
        FieldPtrs);
    std::apply(
        [ArenaSize](auto &...Ptrs) {
          ((Ptrs =
                static_cast<decltype(+Ptrs)>(detail::AllocStorage(ArenaSize))),
           ...);
        },
        BackFieldPtrs);
    PermScratch = static_cast<size_t *>(
        detail::AllocStorage(NumElements * sizeof(size_t)));
    CountPtr = static_cast<AtomicCount *>(
        detail::AllocStorage(sizeof(AtomicCount)));
    // Placement-new the atomic in (already zero-initialized) storage.
    ::new (CountPtr) AtomicCount{0};
  }

  // Host-only destructor body. The shallow-copy path (kernel parameters) lands
  // inside __device__ contexts where cudaFree is unavailable; the device
  // path is deliberately empty. The host-side owning instance is responsible
  // for the eventual free.
  ~SOAAllocator() {
#ifndef __CUDA_ARCH__
    if (!Owns)
      return;
    if (CountPtr)
      CountPtr->~AtomicCount();
    detail::FreeStorage(CountPtr, sizeof(AtomicCount));
    size_t ArenaSize = Capacity * MaxFieldSize();
    std::apply(
        [ArenaSize](auto *...Ptrs) {
          ((detail::FreeStorage(Ptrs, ArenaSize)), ...);
        },
        FieldPtrs);
    std::apply(
        [ArenaSize](auto *...Ptrs) {
          ((detail::FreeStorage(Ptrs, ArenaSize)), ...);
        },
        BackFieldPtrs);
    detail::FreeStorage(PermScratch, Capacity * sizeof(size_t));
#endif
  }

  // Shallow copy: shares pointers, does NOT take ownership. Used to hand the
  // allocator into a CUDA kernel by value. Move + assign remain disabled
  // because there's no use case for them in the framework.
  PLASTIX_HD SOAAllocator(const SOAAllocator &Other) noexcept
      : FieldPtrs{Other.FieldPtrs}, BackFieldPtrs{Other.BackFieldPtrs},
        PermScratch{Other.PermScratch}, CountPtr{Other.CountPtr},
        Capacity{Other.Capacity}, Owns{false} {}
  SOAAllocator &operator=(const SOAAllocator &) = delete;
  SOAAllocator(SOAAllocator &&) = delete;
  SOAAllocator &operator=(SOAAllocator &&) = delete;

  PLASTIX_HD AllocId<T> Allocate() {
    size_t Id = CountPtr->fetch_add(1);
    if (Id >= Capacity) {
      CountPtr->fetch_sub(1);
      return static_cast<size_t>(-1);
    }
    std::apply(
        [Id](auto &...Ptrs) {
          ((::new (&Ptrs[Id]) typename Fields::Type()), ...);
        },
        FieldPtrs);
    std::apply(
        [Id](auto &...Ptrs) {
          ((::new (&Ptrs[Id]) typename Fields::Type()), ...);
        },
        BackFieldPtrs);
    return Id;
  };

  std::pair<size_t, size_t> AllocateMany(size_t N) {
    size_t Begin = CountPtr->fetch_add(N);
    if (Begin + N > Capacity) {
      CountPtr->fetch_sub(N);
      return {static_cast<size_t>(-1), static_cast<size_t>(-1)};
    }
    for (size_t Id = Begin; Id < Begin + N; ++Id) {
      std::apply(
          [Id](auto &...Ptrs) {
            ((::new (&Ptrs[Id]) typename Fields::Type()), ...);
          },
          FieldPtrs);
      std::apply(
          [Id](auto &...Ptrs) {
            ((::new (&Ptrs[Id]) typename Fields::Type()), ...);
          },
          BackFieldPtrs);
    }
    return {Begin, Begin + N};
  }

  template <typename FieldTag> PLASTIX_HD auto &Get(AllocId<T> Id) {
    constexpr auto Index = IndexOf<FieldTag>();
    const auto &Field = std::get<Index>(FieldPtrs);
    return Field[Id];
  };

  template <typename FieldTag> PLASTIX_HD const auto &Get(AllocId<T> Id) const {
    constexpr auto Index = IndexOf<FieldTag>();
    const auto &Field = std::get<Index>(FieldPtrs);
    return Field[Id];
  };

  PLASTIX_HD auto &Get(AllocId<T> Id)
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  };

  PLASTIX_HD const auto &Get(AllocId<T> Id) const
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  };

  template <typename FieldTag> PLASTIX_HD auto *GetArrayFor() {
    constexpr auto Index = IndexOf<FieldTag>();
    return std::get<Index>(FieldPtrs);
  }

  template <typename FieldTag> PLASTIX_HD const auto *GetArrayFor() const {
    constexpr auto Index = IndexOf<FieldTag>();
    return std::get<Index>(FieldPtrs);
  }

  PLASTIX_HD size_t Size() const { return CountPtr->load(); }
  PLASTIX_HD size_t GetCapacity() const { return Capacity; }

  PLASTIX_HD size_t *PermutationScratch() { return PermScratch; }

  void Gather(size_t N) {
    auto GatherField = [this, N](auto *Src, auto *Dst) {
      for (size_t I = 0; I < N; ++I)
        Dst[I] = Src[PermScratch[I]];
    };
    []<size_t... Is>(std::index_sequence<Is...>, auto &Primary, auto &Back,
                     auto &&Fn) {
      ((Fn(std::get<Is>(Primary), std::get<Is>(Back))), ...);
    }(std::index_sequence_for<Fields...>{}, FieldPtrs, BackFieldPtrs,
      GatherField);
    std::swap(FieldPtrs, BackFieldPtrs);
  }
};

} // namespace alloc

// Free-function shorthand for Alloc.template Get<Tag>(Id). Avoids the
// `.template` disambiguator when the allocator is a dependent type (e.g.
// `auto &` policy parameters) and reads a bit cleaner at call sites.
// Const-propagates via auto& deduction.
template <typename Tag>
PLASTIX_HD constexpr auto &GetField(auto &Alloc, size_t Id) {
  return Alloc.template Get<Tag>(Id);
}

} // namespace plastix

#endif // PLASTIX_ALLOC_HPP
