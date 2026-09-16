#ifndef PLASTIX_CONN_HPP
#define PLASTIX_CONN_HPP

#include "plastix/alloc.hpp"
#include "plastix/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace plastix {

struct ConnectionState {};
using ConnStateId = alloc::AllocId<ConnectionState>;

// Core tags -- always present in every connection allocator.
struct FromIdTag {};
struct ToIdTag {};
struct DeadTag {};
struct SrcLevelTag {};

// Convenience tag for the common Weight field (user-provided, not core).
struct WeightTag {};

// Convenience accessor for the common Weight field.
PLASTIX_HD constexpr auto &GetWeight(auto &Alloc, size_t Id) {
  return GetField<WeightTag>(Alloc, Id);
}

// Packs a (From, To) conn-id pair into 64 bits for the device AddConnections
// pipeline (sort/dedup as raw uint64_t). Here so dispatch headers see the type.
struct CompactEdge {
  uint64_t Bits;

  PLASTIX_HD CompactEdge() : Bits(0) {}
  PLASTIX_HD CompactEdge(uint32_t From, uint32_t To)
      : Bits(static_cast<uint64_t>(From) | (static_cast<uint64_t>(To) << 32)) {}

  PLASTIX_HD uint32_t From() const { return static_cast<uint32_t>(Bits); }
  PLASTIX_HD uint32_t To() const { return static_cast<uint32_t>(Bits >> 32); }
};

// Type list for user-defined extra connection fields.
template <typename... Fields> struct ConnFieldList {};

// Concatenate two ConnFieldLists so a traits class can combine its own fields
// with a library bundle's (see WithOptimizer) instead of restating both.
template <typename A, typename B> struct ConcatConnFieldsImpl;

template <typename... A, typename... B>
struct ConcatConnFieldsImpl<ConnFieldList<A...>, ConnFieldList<B...>> {
  using type = ConnFieldList<A..., B...>;
};

template <typename A, typename B>
using ConcatConnFields = typename ConcatConnFieldsImpl<A, B>::type;

// True when no two fields share a tag. Tags are the allocator's only key
// (SOAAllocator::IndexOf returns the first match), so a duplicate would
// silently alias two columns onto one array rather than fail.
template <typename List> struct ConnFieldTagsAreUnique;

template <typename... Fields>
struct ConnFieldTagsAreUnique<ConnFieldList<Fields...>> {
private:
  template <typename Tag> static constexpr size_t CountOf() {
    return ((std::is_same_v<Tag, typename Fields::Tag> ? 1u : 0u) + ...);
  }

public:
  static constexpr bool value =
      ((CountOf<typename Fields::Tag>() == 1u) && ...);
};

template <typename List>
inline constexpr bool ConnFieldTagsAreUniqueV =
    ConnFieldTagsAreUnique<List>::value;

// Connection allocator parameterized by extra fields; core fields (FromId,
// ToId, Dead, SrcLevel) are always present, ExtraFields... are user SOAField
// entries.
// Allocator-template-parameterized form: Alloc is SOAAllocator (default) or the
// device-resident allocator (ADR-006), selected by traits. Both share the
// <Entity, Fields...> signature, so they are interchangeable here.
template <template <class, class...> class Alloc, typename... ExtraFields>
using MakeConnAllocatorWith =
    Alloc<ConnectionState, alloc::SOAField<FromIdTag, uint32_t>,
          alloc::SOAField<ToIdTag, uint32_t>, alloc::SOAField<DeadTag, bool>,
          alloc::SOAField<SrcLevelTag, uint16_t>, ExtraFields...>;

template <typename... ExtraFields>
using MakeConnAllocator =
    MakeConnAllocatorWith<alloc::SOAAllocator, ExtraFields...>;

// Helper to unpack a ConnFieldList into MakeConnAllocatorWith.
template <template <class, class...> class Alloc, typename FL>
struct MakeConnAllocatorFromListWith;

template <template <class, class...> class Alloc, typename... Extra>
struct MakeConnAllocatorFromListWith<Alloc, ConnFieldList<Extra...>> {
  using type = MakeConnAllocatorWith<Alloc, Extra...>;
};

template <template <class, class...> class Alloc, typename FL>
using MakeConnAllocatorFromWith =
    typename MakeConnAllocatorFromListWith<Alloc, FL>::type;

template <typename FL>
using MakeConnAllocatorFrom = MakeConnAllocatorFromWith<alloc::SOAAllocator, FL>;

// Convenience alias for the default case (includes Weight).
using ConnStateAllocator = MakeConnAllocator<alloc::SOAField<WeightTag, float>>;

} // namespace plastix

#endif // PLASTIX_CONN_HPP
