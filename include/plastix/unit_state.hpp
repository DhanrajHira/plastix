#ifndef PLASTIX_UNIT_STATE_HPP
#define PLASTIX_UNIT_STATE_HPP

#include "plastix/alloc.hpp"
#include "plastix/position.hpp"

namespace plastix {

struct UnitState {};
using UnitStateId = alloc::AllocId<UnitState>;

struct ActivationTag {};
struct ForwardAccTag {};
struct BackwardAccTag {};
struct UpdateAccTag {};
struct PrunedTag {};
struct PositionTag {};
struct LevelTag {};

template <typename FwdAcc, typename BwdAcc, typename UpdAcc>
using MakeUnitAllocator =
    alloc::SOAAllocator<UnitState, alloc::SOAField<ActivationTag, float>,
                        alloc::SOAField<ForwardAccTag, FwdAcc>,
                        alloc::SOAField<BackwardAccTag, BwdAcc>,
                        alloc::SOAField<UpdateAccTag, UpdAcc>,
                        alloc::SOAField<PrunedTag, bool>,
                        alloc::SOAField<PositionTag, UnitPosition>,
                        alloc::SOAField<LevelTag, uint16_t>>;

using UnitStateAllocator = MakeUnitAllocator<float, float, float>;

} // namespace plastix

#endif // PLASTIX_UNIT_STATE_HPP
