#ifndef PLASTIX_PLASTIX_HPP
#define PLASTIX_PLASTIX_HPP

#include "plastix/alloc.hpp"
namespace plastix {

/// Returns the library version as a string.
const char *version();

class UnitState {};

template <typename ForwardPassPolicy, typename BackwardPassPolicy,
          typename UpdateUnitStatePolicy, typename UpdateConnStatePolicy,
          typename PruneUnitPolicy, typename PruneConnectionPolicy,
          typename AddUnitPolicy>
class Network {
public:
  void DoForwardPass() {
    ForwardPassPolicy::AccumulateForward();
    ForwardPassPolicy::CalculateAndApplyForward();
  }
  void DoBackwardPass() {
    BackwardPassPolicy::AccumulateBackward();
    BackwardPassPolicy::CalculateAndApplyBackward();
  }
  void DoUpdateUnitState() {
    auto &Mapped = UpdateUnitStatePolicy::Map();
    auto &Combined = UpdateUnitStatePolicy::Combine(Mapped, Mapped);
    UpdateUnitStatePolicy::Apply(Combined);
  }
  void DoUpdateConnectionState() {
    auto &Mapped = UpdateConnStatePolicy::Map();
    auto &Combined = UpdateConnStatePolicy::Combine(Mapped, Mapped);
    UpdateConnStatePolicy::Apply(Combined);
  }
  void DoPruneUnits() { PruneUnitPolicy::ShouldPruneUnit(); }
  void DoPruneConnections() { PruneConnectionPolicy::ShouldPruneConnection(); }
  void DoAddUnits() {}
  void DoAddConnections() {}

private:
  plastix::alloc::SOAAllocator<UnitState> UnitAlloc;
};

} // namespace plastix

#endif // PLASTIX_PLASTIX_HPP
