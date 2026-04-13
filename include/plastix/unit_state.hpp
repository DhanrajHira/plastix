#ifndef PLASTIX_UNIT_STATE_HPP
#define PLASTIX_UNIT_STATE_HPP

#include "plastix/alloc.hpp"

namespace plastix {

struct UnitState {};
using UnitStateId = alloc::AllocId<UnitState>;

struct ActivationTag {};
struct ForwardAccTag {};
struct BackwardAccTag {};
struct PrunedTag {};
struct LevelTag {};

// Convenience accessors for built-in unit fields.
constexpr auto &GetActivation(auto &Alloc, size_t Id) {
  return GetField<ActivationTag>(Alloc, Id);
}
constexpr auto &GetForwardAcc(auto &Alloc, size_t Id) {
  return GetField<ForwardAccTag>(Alloc, Id);
}
constexpr auto &GetBackwardAcc(auto &Alloc, size_t Id) {
  return GetField<BackwardAccTag>(Alloc, Id);
}
constexpr auto &GetLevel(auto &Alloc, size_t Id) {
  return GetField<LevelTag>(Alloc, Id);
}

// Type list for user-defined extra unit fields.
template <typename... Fields> struct UnitFieldList {};

// Unit allocator parameterized by accumulator types from policies.
// FwdAcc, BwdAcc are the Accumulator types from the
// ForwardPass and BackwardPass policies respectively.
// ExtraFields... are additional SOAField<Tag, Type> entries from the user.
template <typename FwdAcc, typename BwdAcc, typename... ExtraFields>
using MakeUnitAllocator =
    alloc::SOAAllocator<UnitState, alloc::SOAField<ActivationTag, float>,
                        alloc::SOAField<ForwardAccTag, FwdAcc>,
                        alloc::SOAField<BackwardAccTag, BwdAcc>,
                        alloc::SOAField<PrunedTag, bool>,
                        alloc::SOAField<LevelTag, uint16_t>, ExtraFields...>;

// Helper to unpack a UnitFieldList into MakeUnitAllocator.
template <typename FwdAcc, typename BwdAcc, typename FL>
struct MakeUnitAllocatorFromList;

template <typename FwdAcc, typename BwdAcc, typename... Extra>
struct MakeUnitAllocatorFromList<FwdAcc, BwdAcc, UnitFieldList<Extra...>> {
  using type = MakeUnitAllocator<FwdAcc, BwdAcc, Extra...>;
};

template <typename FwdAcc, typename BwdAcc, typename FL>
using MakeUnitAllocatorFrom =
    typename MakeUnitAllocatorFromList<FwdAcc, BwdAcc, FL>::type;

// Convenience alias for the default case (all float accumulators, no extras).
using UnitStateAllocator = MakeUnitAllocator<float, float>;

} // namespace plastix

#endif // PLASTIX_UNIT_STATE_HPP
