#ifndef PLASTIX_MACROS_HPP
#define PLASTIX_MACROS_HPP

// PLASTIX_HD: marks a function callable from both host and device code. Every
// accessor/policy/kernel-called utility needs it, else a link error at
// instantiation.
#if defined(__CUDACC__)
#define PLASTIX_HD __host__ __device__ __forceinline__
#define PLASTIX_DEVICE __device__ __forceinline__
#define PLASTIX_HOST __host__ inline
#else
#define PLASTIX_HD inline
#define PLASTIX_DEVICE inline
#define PLASTIX_HOST inline
#endif

#ifdef PLASTIX_HAS_CUDA
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define PLASTIX_CUDA_CHECK(stmt)                                               \
  do {                                                                         \
    cudaError_t Err__ = (stmt);                                                \
    if (Err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(Err__));                                 \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define PLASTIX_CUDA_CHECK_KERNEL() PLASTIX_CUDA_CHECK(cudaGetLastError())
#endif

#endif // PLASTIX_MACROS_HPP
