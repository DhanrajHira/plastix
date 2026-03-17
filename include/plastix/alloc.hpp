#ifndef PLASTIX_ALLOC_HPP
#define PLASTIX_ALLOC_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <new>
#include <sys/mman.h>
#include <tuple>

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
      : FieldPtrs{}, Count{0}, Capacity{NumElements} {
    constexpr size_t MaxFieldSize =
        std::max({sizeof(typename Fields::Type)...});
    size_t ArenaSize = NumElements * MaxFieldSize;
    std::apply(
        [ArenaSize](auto &...Ptrs) {
          ((Ptrs = static_cast<decltype(+Ptrs)>(
                mmap(nullptr, ArenaSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0))),
           ...);
        },
        FieldPtrs);
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
};

// ---------------------------------------------------------------------------
// Page-structured allocation
// ---------------------------------------------------------------------------

template <typename T>
concept PageType = requires(T &P, const T &CP, size_t I) {
  P.WriteSlot(I);
  CP.GetSlot(I);
};

template <typename T, size_t SlotSize> struct Page {
  std::array<T, SlotSize> Slots;
  const T &GetSlot(size_t I) const { return Slots[I]; }
  T &WriteSlot(size_t I) { return Slots[I]; }
};

template <typename Entity, typename... Fields>
  requires(PageType<typename Fields::Type> && ...)
class PageAllocator : public SOAAllocator<Entity, Fields...> {
  using Base = SOAAllocator<Entity, Fields...>;

  template <typename Field>
  void ScatterField(AllocId<Entity> PageId, uint32_t LiveMask) {
    auto &Page = Base::template Get<typename Field::Tag>(PageId);
    uint32_t Remaining = LiveMask;
    size_t Dst = 0;
    while (Remaining) {
      size_t Src = std::countr_zero(Remaining);
      if (Dst != Src)
        Page.WriteSlot(Dst) = Page.GetSlot(Src);
      Remaining &= Remaining - 1; // clear lowest set bit
      ++Dst;
    }
  }

public:
  using Base::Base;

  void CompactPage(AllocId<Entity> PageId, uint32_t LiveMask) {
    (ScatterField<Fields>(PageId, LiveMask), ...);
  }
};

} // namespace alloc

} // namespace plastix

#endif // PLASTIX_ALLOC_HPP
