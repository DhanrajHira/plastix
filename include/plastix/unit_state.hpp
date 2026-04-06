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

// Type list for user-defined extra unit fields.
template <typename... Fields> struct UnitFieldList {};

// Unit allocator parameterized by accumulator types from policies.
// FwdAcc, BwdAcc, UpdAcc are the Accumulator/Partial types from the
// ForwardPass, BackwardPass, and UpdateUnit policies respectively.
// ExtraFields... are additional SOAField<Tag, Type> entries from the user.
template <typename FwdAcc, typename BwdAcc, typename UpdAcc,
          typename... ExtraFields>
using MakeUnitAllocator =
    alloc::SOAAllocator<UnitState, alloc::SOAField<ActivationTag, float>,
                        alloc::SOAField<ForwardAccTag, FwdAcc>,
                        alloc::SOAField<BackwardAccTag, BwdAcc>,
                        alloc::SOAField<UpdateAccTag, UpdAcc>,
                        alloc::SOAField<PrunedTag, bool>, ExtraFields...>;

// Helper to unpack a UnitFieldList into MakeUnitAllocator.
template <typename FwdAcc, typename BwdAcc, typename UpdAcc, typename FL>
struct MakeUnitAllocatorFromList;

template <typename FwdAcc, typename BwdAcc, typename UpdAcc,
          typename... Extra>
struct MakeUnitAllocatorFromList<FwdAcc, BwdAcc, UpdAcc,
                                UnitFieldList<Extra...>> {
  using type = MakeUnitAllocator<FwdAcc, BwdAcc, UpdAcc, Extra...>;
};

template <typename FwdAcc, typename BwdAcc, typename UpdAcc, typename FL>
using MakeUnitAllocatorFrom =
    typename MakeUnitAllocatorFromList<FwdAcc, BwdAcc, UpdAcc, FL>::type;

// Convenience alias for the default case (all float accumulators, no extras).
using UnitStateAllocator = MakeUnitAllocator<float, float, float>;

} // namespace plastix

#endif // PLASTIX_UNIT_STATE_HPP
