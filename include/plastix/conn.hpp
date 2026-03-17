#ifndef PLASTIX_CONN_HPP
#define PLASTIX_CONN_HPP

#include "plastix/alloc.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace plastix {

constexpr static size_t ConnPageSlotSize = 7;

struct ConnPage {
  uint32_t ToUnitIdx;
  uint32_t Count;
  std::array<std::pair<uint32_t, float>, ConnPageSlotSize> Conn;

  const auto &GetSlot(size_t I) const { return Conn[I]; }
  auto &WriteSlot(size_t I) { return Conn[I]; }
};

struct ConnectionState {};
struct ConnPageMarker {};

using ConnStateAllocator =
    alloc::PageAllocator<ConnectionState,
                         alloc::SOAField<ConnPageMarker, ConnPage>>;

} // namespace plastix

#endif // PLASTIX_CONN_HPP
