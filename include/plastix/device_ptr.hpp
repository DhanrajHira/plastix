#ifndef PLASTIX_DEVICE_PTR_HPP
#define PLASTIX_DEVICE_PTR_HPP

// A pointer into device-resident SOA storage (ADR-006). Under the CUDA build the
// dereference operators are __device__-only, so any HOST dereference is a compile
// error (calling a __device__ function from host code) -- turning "a host pass
// forgot this is device memory" into a build failure, not a segfault. On the CPU
// build it is a transparent pointer. get() is the explicit escape hatch for
// allocation / free / cudaMemcpy bookkeeping; it never dereferences.

#include "plastix/macros.hpp"

#include <cstddef>

namespace plastix {

template <typename T> class device_ptr {
  T *Ptr_ = nullptr;

public:
  device_ptr() = default;
  PLASTIX_HD explicit device_ptr(T *P) : Ptr_(P) {}

  // Bookkeeping only -- host may hold/free/copy the address, never deref it.
  PLASTIX_HD T *get() const { return Ptr_; }
  PLASTIX_HD explicit operator bool() const { return Ptr_ != nullptr; }

#if defined(__CUDACC__)
  // Device build: dereference is device-only. Host use fails to compile.
  __device__ T &operator*() const { return *Ptr_; }
  __device__ T &operator[](std::size_t I) const { return Ptr_[I]; }
  __device__ T *operator->() const { return Ptr_; }
#else
  // Host (CPU) build: transparent pointer.
  T &operator*() const { return *Ptr_; }
  T &operator[](std::size_t I) const { return Ptr_[I]; }
  T *operator->() const { return Ptr_; }
#endif
};

} // namespace plastix

#endif // PLASTIX_DEVICE_PTR_HPP
