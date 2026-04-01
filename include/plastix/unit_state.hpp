#ifndef PLASTIX_UNIT_STATE_HPP
#define PLASTIX_UNIT_STATE_HPP

#include "plastix/alloc.hpp"

namespace plastix {

// Entity type and ID
struct UnitState {};
using UnitStateId = alloc::AllocId<UnitState>;

// SOA field tags
struct ActivationTag {};
struct ForwardAccTag {};
struct BackwardAccTag {};
struct UpdateAccTag {};
struct PrunedTag {};

// Unit allocator parameterized by accumulator types from policies.
// FwdAcc, BwdAcc, UpdAcc are the Accumulator/Partial types from the
// ForwardPass, BackwardPass, and UpdateUnit policies respectively.
template <typename FwdAcc, typename BwdAcc, typename UpdAcc>
using MakeUnitAllocator =
    alloc::SOAAllocator<UnitState, alloc::SOAField<ActivationTag, float>,
                        alloc::SOAField<ForwardAccTag, FwdAcc>,
                        alloc::SOAField<BackwardAccTag, BwdAcc>,
                        alloc::SOAField<UpdateAccTag, UpdAcc>,
                        alloc::SOAField<PrunedTag, bool>>;

// Convenience alias for the default case (all float accumulators).
using UnitStateAllocator = MakeUnitAllocator<float, float, float>;

} // namespace plastix

#endif // PLASTIX_UNIT_STATE_HPP
