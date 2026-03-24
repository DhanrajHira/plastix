#ifndef PLASTIX_UNIT_STATE_HPP
#define PLASTIX_UNIT_STATE_HPP

#include "plastix/alloc.hpp"

namespace plastix {

// Generates UnitState tags, UnitStateAllocator, and UnitStateHandle from the
// field list in unit_state.inc by including soa.hpp once per mode.
// See soa.hpp for a full explanation of the three-pass X-macro pattern.

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
