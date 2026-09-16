#ifndef PLASTIX_ALLOC_HPP
#define PLASTIX_ALLOC_HPP

#include "plastix/macros.hpp"

#include <algorithm>
#include <cstddef>
#include <new>
#include <tuple>
#include <utility>

#ifdef PLASTIX_HAS_CUDA
#include <cuda/std/atomic>
#else
#include <atomic>
#include <sys/mman.h>
#endif

namespace plastix {
namespace alloc {

template <typename T> using AllocId = size_t;

// Atomic backing for Count under CUDA (libcu++ so host + device share it).
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

// TODO(plastix-gpu): switch to explicit cudaMalloc + staging copies; managed
// memory is the MVP backend (identical SOA contract on both backends).
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

// Forward declaration so SoaView can befriend its owning allocator.
template <typename T, typename... Fields> class SOAAllocator;

// Non-owning, device-copyable handle over the SOA arenas: plain pointers only,
// never frees (SOAAllocator owns storage), so it passes into kernels by value.
//
// TODO(plastix-gpu): FieldPtrs/BackFieldPtrs are std::tuples, not guaranteed
// trivially-copyable (kernel params ideally are); swap for a pointer aggregate.
template <typename T, typename... Fields> class SoaView {
  std::tuple<typename Fields::Type *...> FieldPtrs{};
  std::tuple<typename Fields::Type *...> BackFieldPtrs{};
  size_t *PermScratch = nullptr;
  AtomicCount *CountPtr = nullptr;
  size_t Capacity = 0;

  // Resolves FieldTag to its index in the Fields... pack at compile time.
  // Uses a short-circuit fold: increments I for each non-matching tag until
  // the matching one is found, then the && chain stops.
  template <typename FieldTag> constexpr static size_t IndexOf() {
    size_t I = 0;
    ((std::is_same_v<FieldTag, typename Fields::Tag> ? false : (++I, true)) &&
     ...);
    return I;
  }

  friend class SOAAllocator<T, Fields...>;

public:
  SoaView() = default;
  SoaView &operator=(const SoaView &) = default;

  // Explicit host/device copy so the handle passes into kernels by value; the
  // implicit copy would be host-only through the std::tuple members.
  PLASTIX_HD SoaView(const SoaView &Other) noexcept
      : FieldPtrs{Other.FieldPtrs}, BackFieldPtrs{Other.BackFieldPtrs},
        PermScratch{Other.PermScratch}, CountPtr{Other.CountPtr},
        Capacity{Other.Capacity} {}

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
  }

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
  }

  template <typename FieldTag> PLASTIX_HD const auto &Get(AllocId<T> Id) const {
    constexpr auto Index = IndexOf<FieldTag>();
    const auto &Field = std::get<Index>(FieldPtrs);
    return Field[Id];
  }

  PLASTIX_HD auto &Get(AllocId<T> Id)
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  }

  PLASTIX_HD const auto &Get(AllocId<T> Id) const
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  }

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

// Owning RAII allocator: builds the SOA arenas (mmap or cudaMallocManaged),
// exposes a non-owning SoaView, frees on destruction. Move-only (one owner
// frees).
template <typename T, typename... Fields> class SOAAllocator {
  SoaView<T, Fields...> View;

  static constexpr size_t MaxFieldSize() {
    return std::max({sizeof(typename Fields::Type)...});
  }

  void Release() {
    if (View.CountPtr == nullptr)
      return;
    View.CountPtr->~AtomicCount();
    detail::FreeStorage(View.CountPtr, sizeof(AtomicCount));
    size_t ArenaSize = View.Capacity * MaxFieldSize();
    std::apply(
        [ArenaSize](auto *...Ptrs) {
          ((detail::FreeStorage(Ptrs, ArenaSize)), ...);
        },
        View.FieldPtrs);
    std::apply(
        [ArenaSize](auto *...Ptrs) {
          ((detail::FreeStorage(Ptrs, ArenaSize)), ...);
        },
        View.BackFieldPtrs);
    detail::FreeStorage(View.PermScratch, View.Capacity * sizeof(size_t));
  }

public:
  explicit SOAAllocator(size_t NumElements) {
    View.Capacity = NumElements;
    size_t ArenaSize = NumElements * MaxFieldSize();
    std::apply(
        [ArenaSize](auto &...Ptrs) {
          ((Ptrs =
                static_cast<decltype(+Ptrs)>(detail::AllocStorage(ArenaSize))),
           ...);
        },
        View.FieldPtrs);
    std::apply(
        [ArenaSize](auto &...Ptrs) {
          ((Ptrs =
                static_cast<decltype(+Ptrs)>(detail::AllocStorage(ArenaSize))),
           ...);
        },
        View.BackFieldPtrs);
    View.PermScratch = static_cast<size_t *>(
        detail::AllocStorage(NumElements * sizeof(size_t)));
    View.CountPtr =
        static_cast<AtomicCount *>(detail::AllocStorage(sizeof(AtomicCount)));
    // Placement-new the atomic in (already zero-initialized) storage.
    ::new (View.CountPtr) AtomicCount{0};
  }

  ~SOAAllocator() { Release(); }

  SOAAllocator(const SOAAllocator &) = delete;
  SOAAllocator &operator=(const SOAAllocator &) = delete;

  // Move transfers ownership; the moved-from view is marked empty (CountPtr
  // null) so its destructor is a no-op.
  SOAAllocator(SOAAllocator &&Other) noexcept : View{Other.View} {
    Other.View.CountPtr = nullptr;
  }
  SOAAllocator &operator=(SOAAllocator &&Other) noexcept {
    if (this != &Other) {
      Release();
      View = Other.View;
      Other.View.CountPtr = nullptr;
    }
    return *this;
  }

  // Non-owning handle for host code and kernel launches.
  PLASTIX_HD SoaView<T, Fields...> view() const { return View; }

  // Forwarders so existing call sites keep working through the owner.
  PLASTIX_HD AllocId<T> Allocate() { return View.Allocate(); }
  std::pair<size_t, size_t> AllocateMany(size_t N) {
    return View.AllocateMany(N);
  }

  template <typename FieldTag> PLASTIX_HD auto &Get(AllocId<T> Id) {
    return View.template Get<FieldTag>(Id);
  }
  template <typename FieldTag> PLASTIX_HD const auto &Get(AllocId<T> Id) const {
    return View.template Get<FieldTag>(Id);
  }
  PLASTIX_HD auto &Get(AllocId<T> Id)
    requires(sizeof...(Fields) == 1)
  {
    return View.Get(Id);
  }
  PLASTIX_HD const auto &Get(AllocId<T> Id) const
    requires(sizeof...(Fields) == 1)
  {
    return View.Get(Id);
  }

  template <typename FieldTag> PLASTIX_HD auto *GetArrayFor() {
    return View.template GetArrayFor<FieldTag>();
  }
  template <typename FieldTag> PLASTIX_HD const auto *GetArrayFor() const {
    return View.template GetArrayFor<FieldTag>();
  }

  PLASTIX_HD size_t Size() const { return View.Size(); }
  PLASTIX_HD size_t GetCapacity() const { return View.GetCapacity(); }
  PLASTIX_HD size_t *PermutationScratch() { return View.PermutationScratch(); }
  void Gather(size_t N) { View.Gather(N); }
};

} // namespace alloc

// Free-function shorthand for Alloc.template Get<Tag>(Id): avoids the
// `.template` disambiguator for dependent allocator types. Const-propagates via
// auto&.
template <typename Tag>
PLASTIX_HD constexpr auto &GetField(auto &Alloc, size_t Id) {
  return Alloc.template Get<Tag>(Id);
}

} // namespace plastix

#endif // PLASTIX_ALLOC_HPP
