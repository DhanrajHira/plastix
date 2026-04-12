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

// AllocId is an integer index into a SOAAllocator array. It is parameterised
// on the entity type only for documentation clarity; there is no runtime
// distinction between ids of different types.
template <typename T> using AllocId = size_t;

// SOAField pairs a unique empty tag type with its data type. The tag is used
// as a compile-time key to locate the right array inside SOAAllocator::Get<>.
// Tag must be an empty class (no data members) so that it carries zero
// runtime cost while remaining a distinct type for each field.
template <typename FieldTag, typename T>

struct SOAField {
  static_assert(std::is_class_v<FieldTag> && std::is_empty_v<FieldTag>,
                "Type FieldTag here is supposed to be an empty marker type. "
                "All the fields that "
                "this type should store should be passed in separately.");
  using Tag = FieldTag;
  using Type = T;
};

// SOAAllocator stores N entities in struct-of-arrays layout: each field has
// its own contiguous array of length Capacity, so iterating all values of a
// single field is cache-friendly. Fields are accessed by tag, which resolves
// to a tuple index at compile time with zero runtime overhead.
//
// Memory is reserved with mmap + MAP_NORESERVE: virtual address space is
// committed up front but physical pages are only faulted in on first access,
// so over-estimating Capacity is cheap.
//
// To add fields beyond the built-in UnitState fields (e.g. for a custom
// learning algorithm), construct a new SOAAllocator<UnitState, ...all base
// fields..., SOAField<MyTag, MyType>> directly — no framework changes needed.
// See examples/ipc-linear/ for a worked example.
template <typename T, typename... Fields> class SOAAllocator {
  std::tuple<typename Fields::Type *...> FieldPtrs;
  std::atomic<size_t> Count;
  size_t Capacity;

  // Resolves FieldTag to its index in the Fields... pack at compile time.
  // Uses a short-circuit fold: increments I for each non-matching tag until
  // the matching one is found, then the && chain stops.
  template <typename FieldTag> constexpr static size_t IndexOf() {
    size_t I = 0;
    ((std::is_same_v<FieldTag, typename Fields::Tag> ? false : (++I, true)) &&
     ...);
    return I;
  }

public:
  explicit SOAAllocator(size_t NumElements)
      : FieldPtrs{}, Count{0}, Capacity{NumElements} {
    // All field arrays are the same length. Size each arena by the largest
    // element type so a single ArenaSize constant covers every field.
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

  // Bump-allocates one entity slot. Uses an atomic increment so the allocator
  // is safe to call from multiple threads during network construction.
  // Returns size_t(-1) on overflow (capacity exceeded).
  // Placement-new value-initialises every field for the new slot.
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

  // Tag-dispatched field access. IndexOf<FieldTag>() is a constexpr that
  // resolves to a tuple index, so std::get<Index> selects the right array
  // and Field[Id] indexes into it — no branching at runtime.
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

  // Convenience overload when there is exactly one field — no tag needed.
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
//
// PageAllocator extends SOAAllocator for entities whose fields are fixed-size
// arrays of slots (PageType). It adds CompactPage() for in-place stream
// compaction, used by DoPruneConnections() to remove dead connections.
// ---------------------------------------------------------------------------

// PageType concept: a field stored in a PageAllocator must expose indexed
// slot access so ScatterField can move slots without knowing the concrete type.
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

  // Compacts one field's slots for a single page in-place.
  // LiveMask is a bitmask where bit i is set if slot i should survive.
  // Surviving slots are moved to the front in index order, overwriting gaps.
  template <typename Field>
  void ScatterField(AllocId<Entity> PageId, uint32_t LiveMask) {
    auto &Page = Base::template Get<typename Field::Tag>(PageId);
    uint32_t Remaining = LiveMask;
    size_t Dst = 0;
    while (Remaining) {
      size_t Src = std::countr_zero(Remaining); // index of lowest set bit
      if (Dst != Src)
        Page.WriteSlot(Dst) = Page.GetSlot(Src);
      Remaining &= Remaining - 1; // clear lowest set bit
      ++Dst;
    }
  }

public:
  using Base::Base;

  // Compacts all fields of one page according to LiveMask. Called by
  // DoPruneConnections() after it has computed which slots survive.
  // The caller is responsible for updating the page's Count field.
  void CompactPage(AllocId<Entity> PageId, uint32_t LiveMask) {
    (ScatterField<Fields>(PageId, LiveMask), ...);
  }
};

} // namespace alloc

} // namespace plastix

#endif // PLASTIX_ALLOC_HPP
