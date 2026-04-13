#ifndef PLASTIX_CONN_HPP
#define PLASTIX_CONN_HPP

#include "plastix/alloc.hpp"

namespace plastix {

struct ConnectionState {};
using ConnStateId = alloc::AllocId<ConnectionState>;

// Core tags — always present in every connection allocator.
struct FromIdTag {};
struct ToIdTag {};
struct DeadTag {};
struct SrcLevelTag {};

// Convenience tag for the common Weight field (user-provided, not core).
struct WeightTag {};

// Convenience accessor for the common Weight field.
constexpr auto &GetWeight(auto &Alloc, size_t Id) {
  return GetField<WeightTag>(Alloc, Id);
}

// Type list for user-defined extra connection fields.
template <typename... Fields> struct ConnFieldList {};

// Connection allocator parameterized by extra fields.
// Core fields (FromId, ToId, Dead, SrcLevel) are always present.
// ExtraFields... are additional SOAField<Tag, Type> entries from the user.
template <typename... ExtraFields>
using MakeConnAllocator =
    alloc::SOAAllocator<ConnectionState, alloc::SOAField<FromIdTag, uint32_t>,
                        alloc::SOAField<ToIdTag, uint32_t>,
                        alloc::SOAField<DeadTag, bool>,
                        alloc::SOAField<SrcLevelTag, uint16_t>, ExtraFields...>;

// Helper to unpack a ConnFieldList into MakeConnAllocator.
template <typename FL> struct MakeConnAllocatorFromList;

template <typename... Extra>
struct MakeConnAllocatorFromList<ConnFieldList<Extra...>> {
  using type = MakeConnAllocator<Extra...>;
};

template <typename FL>
using MakeConnAllocatorFrom = typename MakeConnAllocatorFromList<FL>::type;

// Convenience alias for the default case (includes Weight).
using ConnStateAllocator = MakeConnAllocator<alloc::SOAField<WeightTag, float>>;

} // namespace plastix

#endif // PLASTIX_CONN_HPP
