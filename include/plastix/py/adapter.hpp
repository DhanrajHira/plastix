#ifndef PLASTIX_PY_ADAPTER_HPP
#define PLASTIX_PY_ADAPTER_HPP

// Adapter<Net>: implements NetworkHandle over any Net exposing DoStep / GetOutput
// / GetConnAlloc / GetStep (e.g. plastix::Network via a wrapper, or DeepRNetwork).
// num_input/num_output come from the shim since Network does not expose them
// uniformly. Instantiating this is the only place the user's policies meet the
// compiler, so it is the only nvcc TU on the GPU build (ADR-007).

#include "plastix/py/handle.hpp"

#include <plastix/plastix.hpp>

#include <type_traits>
#include <utility>

namespace plastix::py {

template <class Net> class Adapter : public NetworkHandle {
  Net Net_;
  size_t NIn_, NOut_;

  auto &Conn() { return Net_.GetConnAlloc(); }
  using ConnAllocT = std::remove_cvref_t<decltype(std::declval<Net &>().GetConnAlloc())>;

public:
  template <class... A>
  Adapter(size_t NIn, size_t NOut, A &&...Args)
      : Net_(std::forward<A>(Args)...), NIn_(NIn), NOut_(NOut) {}

  size_t num_input() const override { return NIn_; }
  size_t num_output() const override { return NOut_; }
  size_t step_count() const override { return Net_.GetStep(); }
  bool device_resident() const override {
    return alloc::IsDeviceResidentV<ConnAllocT>;
  }

  void step(std::span<const float> X, std::span<const float> Y) override {
    Net_.DoStep(X, Y);
  }

  std::span<const float> output() override { return Net_.GetOutput(); }

  size_t num_live_conns() override {
    auto &C = Conn();
    using CA = std::remove_cvref_t<decltype(C)>;
    if constexpr (alloc::IsDeviceResidentV<CA>)
      C.template PullFieldSlice<plastix::DeadTag>(0, C.Size());
    size_t N = 0;
    for (size_t K = 0; K < C.Size(); ++K)
      if (!plastix::GetField<plastix::DeadTag>(C, K))
        ++N;
    return N;
  }
};

} // namespace plastix::py

#endif // PLASTIX_PY_ADAPTER_HPP
