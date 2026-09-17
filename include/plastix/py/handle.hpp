#ifndef PLASTIX_PY_HANDLE_HPP
#define PLASTIX_PY_HANDLE_HPP

// Non-template, nanobind-free interface to a user network (ADR-007). Adapter<Net>
// implements it over any Net with the DeepR-style step surface; the Python
// module binds only this handle, so the same binding serves every user network.

#include <cstddef>
#include <span>

namespace plastix::py {

struct NetworkHandle {
  virtual ~NetworkHandle() = default;

  virtual size_t num_input() const = 0;
  virtual size_t num_output() const = 0;
  virtual size_t step_count() const = 0;
  virtual bool device_resident() const = 0;

  // One online step on host inputs/targets (y may be empty for inference).
  virtual void step(std::span<const float> x, std::span<const float> y) = 0;
  // Output activations on the host (length num_output()).
  virtual std::span<const float> output() = 0;
  // Live (non-dead) connection count -- the DeepR trajectory metric.
  virtual size_t num_live_conns() = 0;

  // Zero-copy device path (device-resident CUDA build only; throws otherwise):
  // x/y are device pointers (e.g. torch CUDA tensors), output_device() returns a
  // device pointer to the output activations (length num_output()).
  virtual void step_device(const float *dx, const float *dy) = 0;
  virtual const float *output_device() = 0;
};

} // namespace plastix::py

#endif // PLASTIX_PY_HANDLE_HPP
