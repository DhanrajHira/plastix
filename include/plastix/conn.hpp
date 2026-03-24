#ifndef PLASTIX_CONN_HPP
#define PLASTIX_CONN_HPP

#include "plastix/alloc.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace plastix {

// Each ConnPage holds up to ConnPageSlotSize incoming connections for one
// destination unit. With 7 slots: 2×4-byte header + 7×8-byte (u32, float)
// pairs = 8 + 56 = 64 bytes — exactly one cache line. A unit with more than
// ConnPageSlotSize incoming connections spans multiple consecutive pages, all
// sharing the same ToUnitIdx. The forward pass detects destination changes to
// flush the accumulator between pages.
constexpr static size_t ConnPageSlotSize = 7;

struct ConnPage {
  uint32_t ToUnitIdx; // destination unit index for all slots in this page
  uint32_t Count;     // number of valid (src, weight) pairs in Conn
  std::array<std::pair<uint32_t, float>, ConnPageSlotSize> Conn; // (src id, weight)

  const auto &GetSlot(size_t I) const { return Conn[I]; }
  auto &WriteSlot(size_t I) { return Conn[I]; }
};

// ConnectionState and ConnPageMarker are empty tag types. ConnectionState is
// the entity type for the allocator; ConnPageMarker is the SOA field tag used
// to retrieve ConnPage data via ConnAlloc.Get<ConnPageMarker>(pageId).
struct ConnectionState {};
struct ConnPageMarker {};

using ConnStateAllocator =
    alloc::PageAllocator<ConnectionState,
                         alloc::SOAField<ConnPageMarker, ConnPage>>;

} // namespace plastix

#endif // PLASTIX_CONN_HPP
