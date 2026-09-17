#ifndef PLASTIX_UNIT_STATE_HPP
#define PLASTIX_UNIT_STATE_HPP

#include "plastix/alloc.hpp"
#include "plastix/macros.hpp"

namespace plastix {

struct UnitState {};
using UnitStateId = alloc::AllocId<UnitState>;

struct ActivationTag {};
struct ForwardAccTag {};
struct BackwardAccTag {};
struct PrunedTag {};
struct LevelTag {};

// Convenience accessors for built-in unit fields.
PLASTIX_HD constexpr auto &GetActivation(auto &Alloc, size_t Id) {
  return GetField<ActivationTag>(Alloc, Id);
}
PLASTIX_HD constexpr auto &GetForwardAcc(auto &Alloc, size_t Id) {
  return GetField<ForwardAccTag>(Alloc, Id);
}
PLASTIX_HD constexpr auto &GetBackwardAcc(auto &Alloc, size_t Id) {
  return GetField<BackwardAccTag>(Alloc, Id);
}
PLASTIX_HD constexpr auto &GetLevel(auto &Alloc, size_t Id) {
  return GetField<LevelTag>(Alloc, Id);
}

// Type list for user-defined extra unit fields.
template <typename... Fields> struct UnitFieldList {};

// Unit allocator parameterized by policy accumulators: FwdAcc/BwdAcc from
// Forward/BackwardPass; ExtraFields... are user SOAField<Tag, Type> entries.
// Allocator-template-parameterized form (see conn.hpp MakeConnAllocatorWith):
// Alloc is SOAAllocator (default) or the device-resident allocator (ADR-006).
template <template <class, class...> class Alloc, typename FwdAcc,
          typename BwdAcc, typename... ExtraFields>
using MakeUnitAllocatorWith =
    Alloc<UnitState, alloc::SOAField<ActivationTag, float>,
          alloc::SOAField<ForwardAccTag, FwdAcc>,
          alloc::SOAField<BackwardAccTag, BwdAcc>,
          alloc::SOAField<PrunedTag, bool>,
          alloc::SOAField<LevelTag, uint16_t>, ExtraFields...>;

template <typename FwdAcc, typename BwdAcc, typename... ExtraFields>
using MakeUnitAllocator =
    MakeUnitAllocatorWith<alloc::SOAAllocator, FwdAcc, BwdAcc, ExtraFields...>;

// Helper to unpack a UnitFieldList into MakeUnitAllocatorWith.
template <template <class, class...> class Alloc, typename FwdAcc,
          typename BwdAcc, typename FL>
struct MakeUnitAllocatorFromListWith;

template <template <class, class...> class Alloc, typename FwdAcc,
          typename BwdAcc, typename... Extra>
struct MakeUnitAllocatorFromListWith<Alloc, FwdAcc, BwdAcc,
                                     UnitFieldList<Extra...>> {
  using type = MakeUnitAllocatorWith<Alloc, FwdAcc, BwdAcc, Extra...>;
};

template <template <class, class...> class Alloc, typename FwdAcc,
          typename BwdAcc, typename FL>
using MakeUnitAllocatorFromWith =
    typename MakeUnitAllocatorFromListWith<Alloc, FwdAcc, BwdAcc, FL>::type;

template <typename FwdAcc, typename BwdAcc, typename FL>
using MakeUnitAllocatorFrom =
    MakeUnitAllocatorFromWith<alloc::SOAAllocator, FwdAcc, BwdAcc, FL>;

// Convenience alias for the default case (all float accumulators, no extras).
using UnitStateAllocator = MakeUnitAllocator<float, float>;

} // namespace plastix

#endif // PLASTIX_UNIT_STATE_HPP
