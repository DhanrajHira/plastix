#ifndef PLASTIX_MACROS_HPP
#define PLASTIX_MACROS_HPP

// PLASTIX_HD: marks a function as callable from both host and device code.
//
// Under nvcc this expands to `__host__ __device__ __forceinline__` so the
// compiler emits both host and device variants of the function and inlines
// it aggressively (kernels read these in tight loops). Outside nvcc the
// macro collapses to `inline`, which keeps the framework portable to a
// host-only build.
//
// Every accessor, every default/noop policy method, and every framework
// utility called from inside a CUDA kernel must be tagged with this. User
// policy methods follow the same rule: a missing PLASTIX_HD shows up as a
// link-time error when the kernel body is instantiated.
#if defined(__CUDACC__)
#define PLASTIX_HD __host__ __device__ __forceinline__
#define PLASTIX_DEVICE __device__ __forceinline__
#define PLASTIX_HOST __host__ inline
#else
#define PLASTIX_HD inline
#define PLASTIX_DEVICE inline
#define PLASTIX_HOST inline
#endif

#endif // PLASTIX_MACROS_HPP
