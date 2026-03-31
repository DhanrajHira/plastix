#ifndef PLASTIX_ALLOC_HPP
#define PLASTIX_ALLOC_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <sys/mman.h>
#include <tuple>
#include <utility>

namespace plastix {
namespace alloc {
template <typename T> using AllocId = size_t;

template <typename FieldTag, typename T>

struct SOAField {
  static_assert(std::is_class_v<FieldTag> && std::is_empty_v<FieldTag>,
                "Type FieldTag here is supposed to be an empty marker type. "
                "All the fields that "
                "this type should store should be passed in separately.");
  using Tag = FieldTag;
  using Type = T;
};

template <typename T, typename... Fields> class SOAAllocator {
  std::tuple<typename Fields::Type *...> FieldPtrs;
  std::tuple<typename Fields::Type *...> BackFieldPtrs;
  size_t *PermScratch;
  std::atomic<size_t> Count;
  size_t Capacity;

  template <typename FieldTag> constexpr static size_t IndexOf() {
    size_t I = 0;
    ((std::is_same_v<FieldTag, typename Fields::Tag> ? false : (++I, true)) &&
     ...);
    return I;
  }

public:
  explicit SOAAllocator(size_t NumElements)
      : FieldPtrs{}, BackFieldPtrs{}, PermScratch{nullptr}, Count{0},
        Capacity{NumElements} {
    constexpr size_t MaxFieldSize =
        std::max({sizeof(typename Fields::Type)...});
    size_t ArenaSize = NumElements * MaxFieldSize;
    auto MmapArena = [ArenaSize]() {
      return mmap(nullptr, ArenaSize, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    };
    std::apply(
        [&MmapArena](auto &...Ptrs) {
          ((Ptrs = static_cast<decltype(+Ptrs)>(MmapArena())), ...);
        },
        FieldPtrs);
    std::apply(
        [&MmapArena](auto &...Ptrs) {
          ((Ptrs = static_cast<decltype(+Ptrs)>(MmapArena())), ...);
        },
        BackFieldPtrs);
    PermScratch = static_cast<size_t *>(
        mmap(nullptr, NumElements * sizeof(size_t), PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
  }

  AllocId<T> Allocate() {
    size_t Id = Count.fetch_add(1);
    if (Id >= Capacity) {
      Count.fetch_sub(1);
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

  template <typename FieldTag> auto &Get(AllocId<T> Id) {
    constexpr auto Index = IndexOf<FieldTag>();
    const auto &Field = std::get<Index>(FieldPtrs);
    return Field[Id];
  };

  template <typename FieldTag> const auto &Get(AllocId<T> Id) const {
    constexpr auto Index = IndexOf<FieldTag>();
    const auto &Field = std::get<Index>(FieldPtrs);
    return Field[Id];
  };

  auto &Get(AllocId<T> Id)
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  };

  const auto &Get(AllocId<T> Id) const
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  };

  size_t Size() const { return Count.load(); }

  size_t *PermutationScratch() { return PermScratch; }

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

} // namespace plastix

#endif // PLASTIX_ALLOC_HPP
