// Field list for the built-in UnitState SOA type. This file is included
// by unit_state.hpp three times via soa.hpp to generate tag structs, the
// UnitStateAllocator typedef, and the UnitStateHandle accessor struct.
//
// ActivationA / ActivationB — ping-pong activation buffers. The network
//   alternates which buffer is "current" and which is "previous" each step
//   so that all units in a pass read stable values from the prior step.
//
// BackwardAcc — accumulator for the backward pass. Zeroed after each
//   DoBackwardPass() call.
//
// UpdateAcc — accumulator for DoUpdateUnitState(). Holds the reduced
//   Partial value across all incoming connections before Apply() fires.
//   Zeroed after each DoUpdateUnitState() call.
//
// Pruned — set by DoPruneUnits(); read by DoPruneConnections() to remove
//   any connection whose source or destination unit is flagged.
SOA_TYPE(UnitState)
FIELD(ActivationA, float)
FIELD(ActivationB, float)
FIELD(BackwardAcc, float)
FIELD(UpdateAcc, float)
FIELD(Pruned, bool)
FIELD(PositionX, float)
FIELD(PositionY, float)
FIELD(PositionZ, float)
FIELD(RadiusOfInfluence, float)
SOA_END()
