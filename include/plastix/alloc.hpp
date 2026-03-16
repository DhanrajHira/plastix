#ifndef PLASTIX_ALLOC_HPP
#define PLASTIX_ALLOC_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
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
    return Id;
  };

  template <typename FieldTag> auto &Get(AllocId<T> Id) {
    constexpr auto Index = IndexOf<FieldTag>();
    const auto &Field = std::get<Index>(FieldPtrs);
    return Field[Id];
  };

  auto &Get(AllocId<T> Id)
    requires(sizeof...(Fields) == 1)
  {
    return std::get<0>(FieldPtrs)[Id];
  };
};

namespace detail {
template <typename Page> struct PageTag {};
} // namespace detail

template <typename Page>
using PageAllocator = SOAAllocator<Page, SOAField<detail::PageTag<Page>, Page>>;

} // namespace alloc

} // namespace plastix

#endif // PLASTIX_ALLOC_HPP
