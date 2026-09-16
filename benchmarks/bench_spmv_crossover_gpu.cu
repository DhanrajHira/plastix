// GPU counterpart of bench_spmv_crossover.cpp.
//
// Compares Plastix's CUDA forward path (gpu::DoForwardTopological) against
// cuBLAS sgemv (dense) and cuSPARSE SpMV on CSR (sparse) for the same
// (In, Out, density) sweep. Same edge predicate/weight draws as the CPU
// bench so a side-by-side comparison is meaningful.
//
// NOTE: this machine has no usable CUDA drivers, so the file is built only
// when PLASTIX_ENABLE_CUDA=ON. It compiles but has not been run end-to-end.
// Once a machine with a CUDA toolkit + driver is available, configure with
//   cmake -S . -B build-cuda -DPLASTIX_ENABLE_CUDA=ON
//   cmake --build build-cuda --target bench_spmv_crossover_gpu
// and run:
//   build-cuda/benchmarks/bench_spmv_crossover_gpu --benchmark_format=csv \
//       --benchmark_out=spmv_gpu.csv
//
// Args per benchmark: (In, Out, density_bps) — identical to the CPU file.

#include <benchmark/benchmark.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "plastix/plastix.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#define CUDA_OK(expr)                                                          \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA %s @ %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                   cudaGetErrorString(_e));                                    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define CUBLAS_OK(expr)                                                        \
  do {                                                                         \
    cublasStatus_t _e = (expr);                                                \
    if (_e != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS %s @ %s:%d: status=%d\n", #expr, __FILE__,  \
                   __LINE__, static_cast<int>(_e));                            \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define CUSPARSE_OK(expr)                                                      \
  do {                                                                         \
    cusparseStatus_t _e = (expr);                                              \
    if (_e != CUSPARSE_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuSPARSE %s @ %s:%d: status=%d\n", #expr,          \
                   __FILE__, __LINE__, static_cast<int>(_e));                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Same Plastix configuration as the CPU bench so weights / counts match.
// ---------------------------------------------------------------------------

struct IdentityFwd {
  using Accumulator = float;
  PLASTIX_HD static float Map(auto &U, size_t, size_t SrcId, auto &C,
                              size_t ConnId, auto &) {
    return plastix::GetWeight(C, ConnId) * plastix::GetActivation(U, SrcId);
  }
  PLASTIX_HD static float Combine(float A, float B) { return A + B; }
  PLASTIX_HD static void Apply(auto &U, size_t Id, auto &, float Acc) {
    plastix::GetActivation(U, Id) = Acc;
  }
};

struct BenchTraits : plastix::DefaultNetworkTraits<> {
  using ForwardPass = IdentityFwd;
  static constexpr size_t UnitCapacity = 32 * 1024;
  static constexpr size_t ConnCapacity = 16 * 1024 * 1024;
};

using BenchNet = plastix::Network<BenchTraits>;

static constexpr uint64_t kEdgeSeed = 0xC0FFEE5EED5;
static constexpr uint64_t kWeightSeed = 0xDEADBEEFC0DECAFE;
static constexpr uint64_t kInputSeed = 0xABAD1DEA;

static inline bool EdgeExists(size_t Src, size_t Dst, size_t Out,
                              float Density) {
  return plastix::Bernoulli(kEdgeSeed, Src * Out + Dst, Density);
}
static inline float EdgeWeight(size_t Src, size_t Dst, size_t Out) {
  return plastix::UniformReal(kWeightSeed, Src * Out + Dst, -1.0f, 1.0f);
}

struct RandomSparseLayer {
  size_t NumUnits;
  float Density;
  size_t *NnzOut;

  template <typename UnitAlloc, typename ConnAlloc>
  plastix::UnitRange operator()(UnitAlloc &UA, ConnAlloc &CA,
                                plastix::UnitRange Prev) const {
    uint16_t SrcLevel = plastix::GetLevel(UA, Prev.Begin);
    uint16_t NewLevel = SrcLevel + 1;
    plastix::UnitRange Units = UA.AllocateMany(NumUnits);
    for (auto Id : Units.Ids())
      plastix::GetLevel(UA, Id) = NewLevel;

    size_t In = Prev.End - Prev.Begin;
    size_t Out = NumUnits;
    size_t Nnz = 0;
    for (size_t Src = 0; Src < In; ++Src) {
      for (size_t Dst = 0; Dst < Out; ++Dst) {
        if (!EdgeExists(Src, Dst, Out, Density))
          continue;
        auto Cid = CA.Allocate();
        plastix::GetField<plastix::FromIdTag>(CA, Cid) =
            static_cast<uint32_t>(Prev.Begin + Src);
        plastix::GetField<plastix::ToIdTag>(CA, Cid) =
            static_cast<uint32_t>(Units.Begin + Dst);
        plastix::GetField<plastix::SrcLevelTag>(CA, Cid) = SrcLevel;
        plastix::GetWeight(CA, Cid) = EdgeWeight(Src, Dst, Out);
        ++Nnz;
      }
    }
    if (NnzOut)
      *NnzOut = Nnz;
    return Units;
  }
};

// Dense Out x In row-major float buffer matching the same edges/weights.
static std::vector<float> BuildDenseMatrix(size_t In, size_t Out,
                                           float Density, size_t *NnzOut) {
  std::vector<float> W(Out * In, 0.0f);
  size_t Nnz = 0;
  for (size_t Src = 0; Src < In; ++Src) {
    for (size_t Dst = 0; Dst < Out; ++Dst) {
      if (!EdgeExists(Src, Dst, Out, Density))
        continue;
      W[Dst * In + Src] = EdgeWeight(Src, Dst, Out);
      ++Nnz;
    }
  }
  if (NnzOut)
    *NnzOut = Nnz;
  return W;
}

// CSR (row=Out) of the same matrix for cuSPARSE.
struct Csr {
  std::vector<int> RowPtr; // length Out+1
  std::vector<int> ColIdx; // length Nnz
  std::vector<float> Val;  // length Nnz
};
static Csr BuildCsr(size_t In, size_t Out, float Density, size_t *NnzOut) {
  Csr C;
  C.RowPtr.resize(Out + 1, 0);
  size_t Nnz = 0;
  for (size_t Dst = 0; Dst < Out; ++Dst) {
    C.RowPtr[Dst] = static_cast<int>(Nnz);
    for (size_t Src = 0; Src < In; ++Src) {
      if (!EdgeExists(Src, Dst, Out, Density))
        continue;
      C.ColIdx.push_back(static_cast<int>(Src));
      C.Val.push_back(EdgeWeight(Src, Dst, Out));
      ++Nnz;
    }
  }
  C.RowPtr[Out] = static_cast<int>(Nnz);
  if (NnzOut)
    *NnzOut = Nnz;
  return C;
}

static std::vector<float> RandomVector(size_t N) {
  std::vector<float> V(N);
  for (size_t I = 0; I < N; ++I)
    V[I] = plastix::UniformReal(kInputSeed, I, -1.0f, 1.0f);
  return V;
}

static inline std::tuple<size_t, size_t, float>
ParseArgs(const benchmark::State &S) {
  return {static_cast<size_t>(S.range(0)), static_cast<size_t>(S.range(1)),
          static_cast<float>(S.range(2)) / 10000.0f};
}

static inline void AnnotateCounters(benchmark::State &S, size_t In, size_t Out,
                                    float Density, size_t Nnz) {
  S.counters["In"] = static_cast<double>(In);
  S.counters["Out"] = static_cast<double>(Out);
  S.counters["density"] = Density;
  S.counters["nnz"] = static_cast<double>(Nnz);
  S.counters["dense_macs"] = static_cast<double>(In * Out);
  S.counters["sparse_macs"] = static_cast<double>(Nnz);
}

// ---------------------------------------------------------------------------
// Plastix: build on host, the SOAAllocator's unified-memory path means GPU
// kernels read the same arena. Forward-pass is dispatched to the GPU when
// PLASTIX_HAS_CUDA is defined and Accumulator=float.
// ---------------------------------------------------------------------------

static void BM_Plastix(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  BenchNet Net(In, RandomSparseLayer{Out, Density, &Nnz});
  auto Input = RandomVector(In);
  std::span<const float> InSpan(Input);

  // Warm up once outside the timed loop so the launch-config / level cache
  // are populated.
  Net.DoForwardPass(InSpan);
  CUDA_OK(cudaDeviceSynchronize());

  for (auto _ : S) {
    Net.DoForwardPass(InSpan);
    CUDA_OK(cudaDeviceSynchronize());
    auto Output = Net.GetOutput();
    benchmark::DoNotOptimize(Output.data());
  }
  AnnotateCounters(S, In, Out, Density, Nnz);
}

// ---------------------------------------------------------------------------
// cuBLAS sgemv on the dense W. Row-major source data — cuBLAS is column-major,
// so we pass W as A^T with CUBLAS_OP_T and dimensions (In, Out).
// ---------------------------------------------------------------------------

struct CublasState {
  cublasHandle_t H;
  CublasState() { CUBLAS_OK(cublasCreate(&H)); }
  ~CublasState() { cublasDestroy(H); }
};

static void BM_Cublas(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &Nnz); // row-major Out x In
  auto Input = RandomVector(In);

  float *dW = nullptr, *dX = nullptr, *dY = nullptr;
  CUDA_OK(cudaMalloc(&dW, Out * In * sizeof(float)));
  CUDA_OK(cudaMalloc(&dX, In * sizeof(float)));
  CUDA_OK(cudaMalloc(&dY, Out * sizeof(float)));
  CUDA_OK(cudaMemcpy(dW, W.data(), Out * In * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dX, Input.data(), In * sizeof(float),
                     cudaMemcpyHostToDevice));

  CublasState C;
  const float Alpha = 1.0f, Beta = 0.0f;

  // Warm-up
  CUBLAS_OK(cublasSgemv(C.H, CUBLAS_OP_T, static_cast<int>(In),
                        static_cast<int>(Out), &Alpha, dW, static_cast<int>(In),
                        dX, 1, &Beta, dY, 1));
  CUDA_OK(cudaDeviceSynchronize());

  for (auto _ : S) {
    CUBLAS_OK(cublasSgemv(C.H, CUBLAS_OP_T, static_cast<int>(In),
                          static_cast<int>(Out), &Alpha, dW,
                          static_cast<int>(In), dX, 1, &Beta, dY, 1));
    CUDA_OK(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(dY);
  }

  cudaFree(dW);
  cudaFree(dX);
  cudaFree(dY);
  AnnotateCounters(S, In, Out, Density, Nnz);
}

// ---------------------------------------------------------------------------
// cuSPARSE SpMV on CSR with generic API (cusparseSpMV).
// ---------------------------------------------------------------------------

static void BM_Cusparse(benchmark::State &S) {
  auto [In, Out, Density] = ParseArgs(S);
  size_t Nnz = 0;
  auto Csr = BuildCsr(In, Out, Density, &Nnz);
  auto Input = RandomVector(In);

  int *dRow = nullptr, *dCol = nullptr;
  float *dVal = nullptr, *dX = nullptr, *dY = nullptr;
  CUDA_OK(cudaMalloc(&dRow, Csr.RowPtr.size() * sizeof(int)));
  CUDA_OK(cudaMalloc(&dCol, Csr.ColIdx.size() * sizeof(int)));
  CUDA_OK(cudaMalloc(&dVal, Csr.Val.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&dX, In * sizeof(float)));
  CUDA_OK(cudaMalloc(&dY, Out * sizeof(float)));
  CUDA_OK(cudaMemcpy(dRow, Csr.RowPtr.data(),
                     Csr.RowPtr.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dCol, Csr.ColIdx.data(),
                     Csr.ColIdx.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dVal, Csr.Val.data(), Csr.Val.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dX, Input.data(), In * sizeof(float),
                     cudaMemcpyHostToDevice));

  cusparseHandle_t H;
  CUSPARSE_OK(cusparseCreate(&H));
  cusparseSpMatDescr_t MatA;
  cusparseDnVecDescr_t VecX, VecY;
  CUSPARSE_OK(cusparseCreateCsr(
      &MatA, static_cast<int64_t>(Out), static_cast<int64_t>(In),
      static_cast<int64_t>(Nnz), dRow, dCol, dVal, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CUSPARSE_OK(cusparseCreateDnVec(&VecX, static_cast<int64_t>(In), dX,
                                  CUDA_R_32F));
  CUSPARSE_OK(cusparseCreateDnVec(&VecY, static_cast<int64_t>(Out), dY,
                                  CUDA_R_32F));

  float Alpha = 1.0f, Beta = 0.0f;
  size_t BufBytes = 0;
  CUSPARSE_OK(cusparseSpMV_bufferSize(
      H, CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, MatA, VecX, &Beta, VecY,
      CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &BufBytes));
  void *dBuf = nullptr;
  if (BufBytes > 0)
    CUDA_OK(cudaMalloc(&dBuf, BufBytes));

  // Warm-up
  CUSPARSE_OK(cusparseSpMV(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, MatA,
                           VecX, &Beta, VecY, CUDA_R_32F,
                           CUSPARSE_SPMV_CSR_ALG2, dBuf));
  CUDA_OK(cudaDeviceSynchronize());

  for (auto _ : S) {
    CUSPARSE_OK(cusparseSpMV(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &Alpha, MatA,
                             VecX, &Beta, VecY, CUDA_R_32F,
                             CUSPARSE_SPMV_CSR_ALG2, dBuf));
    CUDA_OK(cudaDeviceSynchronize());
    benchmark::DoNotOptimize(dY);
  }

  if (dBuf)
    cudaFree(dBuf);
  cusparseDestroySpMat(MatA);
  cusparseDestroyDnVec(VecX);
  cusparseDestroyDnVec(VecY);
  cusparseDestroy(H);
  cudaFree(dRow);
  cudaFree(dCol);
  cudaFree(dVal);
  cudaFree(dX);
  cudaFree(dY);
  AnnotateCounters(S, In, Out, Density, Nnz);
}

// ---------------------------------------------------------------------------
// Same correctness check shape as the CPU bench. We trust the CPU bench's
// validation that Plastix == reference, and only check cuBLAS / cuSPARSE
// against the same reference here.
// ---------------------------------------------------------------------------

static void Validate() {
  const size_t In = 64, Out = 32;
  const float Density = 0.5f;
  size_t Nnz = 0;
  auto W = BuildDenseMatrix(In, Out, Density, &Nnz);
  auto Input = RandomVector(In);
  std::vector<float> Ref(Out, 0.0f);
  for (size_t I = 0; I < Out; ++I) {
    float Acc = 0.0f;
    for (size_t J = 0; J < In; ++J)
      Acc += W[I * In + J] * Input[J];
    Ref[I] = Acc;
  }

  // cuBLAS
  {
    float *dW, *dX, *dY;
    CUDA_OK(cudaMalloc(&dW, Out * In * sizeof(float)));
    CUDA_OK(cudaMalloc(&dX, In * sizeof(float)));
    CUDA_OK(cudaMalloc(&dY, Out * sizeof(float)));
    CUDA_OK(cudaMemcpy(dW, W.data(), Out * In * sizeof(float),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dX, Input.data(), In * sizeof(float),
                       cudaMemcpyHostToDevice));
    CublasState C;
    const float A = 1.0f, B = 0.0f;
    CUBLAS_OK(cublasSgemv(C.H, CUBLAS_OP_T, static_cast<int>(In),
                          static_cast<int>(Out), &A, dW, static_cast<int>(In),
                          dX, 1, &B, dY, 1));
    std::vector<float> HOut(Out);
    CUDA_OK(cudaMemcpy(HOut.data(), dY, Out * sizeof(float),
                       cudaMemcpyDeviceToHost));
    for (size_t I = 0; I < Out; ++I) {
      float Tol = 1e-3f * std::max(1.0f, std::abs(Ref[I]));
      if (std::abs(HOut[I] - Ref[I]) > Tol) {
        std::fprintf(stderr, "cuBLAS validation failed at %zu\n", I);
        std::abort();
      }
    }
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY);
  }
  std::fprintf(stderr, "GPU validation OK (cuBLAS).\n");
}

static void RegisterAll() {
  const std::vector<std::pair<int64_t, int64_t>> Shapes = {
      {256, 256},   {1024, 1024}, {4096, 4096},
      {4096, 1024}, {8192, 64},   {64, 8192},
  };
  const std::vector<int64_t> Densities = {10,   100,  200,  500,  1000, 1500,
                                          2000, 2500, 3000, 4000, 5000};

  auto Reg = [](const char *Name, auto Fn, int64_t In, int64_t Out,
                int64_t D) {
    benchmark::RegisterBenchmark(Name, Fn)
        ->Args({In, Out, D})
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.5);
  };

  for (auto [In, Out] : Shapes) {
    for (auto D : Densities) {
      Reg("Plastix", BM_Plastix, In, Out, D);
      Reg("cuBLAS", BM_Cublas, In, Out, D);
      Reg("cuSPARSE", BM_Cusparse, In, Out, D);
    }
  }
}

int main(int argc, char **argv) {
  Validate();
  RegisterAll();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
