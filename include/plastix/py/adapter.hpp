#ifndef PLASTIX_PY_ADAPTER_HPP
#define PLASTIX_PY_ADAPTER_HPP

// Adapter<Net>: implements NetworkHandle over any Net exposing DoStep / GetOutput
// / GetConnAlloc / GetStep (e.g. plastix::Network via a wrapper, or DeepRNetwork).
// num_input/num_output come from the shim since Network does not expose them
// uniformly. Instantiating this is the only place the user's policies meet the
// compiler, so it is the only nvcc TU on the GPU build (ADR-007).

#include "plastix/py/handle.hpp"

#include <plastix/plastix.hpp>

#include <stdexcept>
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

  std::span<const float> output() override {
    // Device-resident nets keep activations on device; refresh host staging so
    // GetOutput() is current. No-op internally on the managed/CPU backend.
    if constexpr (requires { Net_.SyncOutputToHost(); })
      Net_.SyncOutputToHost();
    return Net_.GetOutput();
  }

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

  void step_device(const float *dx, const float *dy) override {
#ifdef PLASTIX_HAS_CUDA
    if constexpr (alloc::IsDeviceResidentV<ConnAllocT>) {
      Net_.DoStepDevice(dx, dy);
      return;
    }
#endif
    (void)dx;
    (void)dy;
    throw std::runtime_error("step_device requires a device-resident CUDA build");
  }

  const float *output_device() override {
#ifdef PLASTIX_HAS_CUDA
    if constexpr (alloc::IsDeviceResidentV<ConnAllocT>)
      return Net_.GetOutputDevice();
#endif
    throw std::runtime_error("output_device requires a device-resident CUDA build");
  }
};

} // namespace plastix::py

#endif // PLASTIX_PY_ADAPTER_HPP
