#ifndef PLASTIX_CONN_HPP
#define PLASTIX_CONN_HPP

#include "plastix/alloc.hpp"
#include "plastix/macros.hpp"

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
PLASTIX_HD constexpr auto &GetWeight(auto &Alloc, size_t Id) {
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

// Packed (From, To) edge key used by the connection-proposal pipeline. Defined
// here (rather than next to the Network) so the GPU dispatch layer
// (dispatch_gpu.hpp) — which is included before the Network and takes
// sizeof(CompactEdge) in a non-dependent context — sees a complete type.
struct CompactEdge {
  uint64_t Bits;

  CompactEdge() : Bits(0) {}
  CompactEdge(uint32_t From, uint32_t To)
      : Bits(static_cast<uint64_t>(From) | (static_cast<uint64_t>(To) << 32)) {}

  uint32_t From() const { return static_cast<uint32_t>(Bits); }
  uint32_t To() const { return static_cast<uint32_t>(Bits >> 32); }
};

} // namespace plastix

#endif // PLASTIX_CONN_HPP
