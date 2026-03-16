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
};

struct ConnectionState {};
struct ConnPageMarker {};

using ConnStateAllocator =
    alloc::SOAAllocator<ConnectionState,
                        alloc::SOAField<ConnPageMarker, ConnPage>>;

} // namespace plastix

#endif // PLASTIX_CONN_HPP
