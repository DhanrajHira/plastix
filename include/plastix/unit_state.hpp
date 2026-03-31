#ifndef PLASTIX_UNIT_STATE_HPP
#define PLASTIX_UNIT_STATE_HPP

#include "plastix/alloc.hpp"
#include "plastix/position.hpp"

namespace plastix {

#define PLASTIX_SOA_MODE_TAGS
#include "plastix/soa.hpp"
#include "unit_state.inc"

#define PLASTIX_SOA_MODE_ALLOC
#include "plastix/soa.hpp"
#include "unit_state.inc"

#define PLASTIX_SOA_MODE_HANDLE
#include "plastix/soa.hpp"
#include "unit_state.inc"

} // namespace plastix

#endif // PLASTIX_UNIT_STATE_HPP
