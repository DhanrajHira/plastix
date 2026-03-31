#ifndef PLASTIX_CONN_HPP
#define PLASTIX_CONN_HPP

#include "plastix/alloc.hpp"

namespace plastix {

#define PLASTIX_SOA_MODE_TAGS
#include "plastix/soa.hpp"
#include "conn_state.inc"

#define PLASTIX_SOA_MODE_ALLOC
#include "plastix/soa.hpp"
#include "conn_state.inc"

#define PLASTIX_SOA_MODE_HANDLE
#include "plastix/soa.hpp"
#include "conn_state.inc"

using ConnStateAllocator = ConnectionStateAllocator;

} // namespace plastix

#endif // PLASTIX_CONN_HPP
