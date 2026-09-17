#ifndef PLASTIX_CUDA_GRAPH_HPP
#define PLASTIX_CUDA_GRAPH_HPP

// RAII capture/instantiate/replay of a CUDA graph (ADR-008 P0): record a
// repeated launch sequence once, replay it with one cudaGraphLaunch. The capture
// body must issue async work only on the capture stream (no sync).

#include "plastix/macros.hpp"

#ifdef PLASTIX_HAS_CUDA

#include <cuda_runtime.h>

namespace plastix {
namespace gpu {

class CudaGraph {
  cudaGraph_t Graph_ = nullptr;
  cudaGraphExec_t Exec_ = nullptr;

public:
  CudaGraph() = default;
  ~CudaGraph() { Reset(); }

  CudaGraph(const CudaGraph &) = delete;
  CudaGraph &operator=(const CudaGraph &) = delete;
  CudaGraph(CudaGraph &&O) noexcept : Graph_(O.Graph_), Exec_(O.Exec_) {
    O.Graph_ = nullptr;
    O.Exec_ = nullptr;
  }
  CudaGraph &operator=(CudaGraph &&O) noexcept {
    if (this != &O) {
      Reset();
      Graph_ = O.Graph_;
      Exec_ = O.Exec_;
      O.Graph_ = nullptr;
      O.Exec_ = nullptr;
    }
    return *this;
  }

  bool ready() const { return Exec_ != nullptr; }

  // Capture Body()'s launches on Stream into an executable graph (discards any
  // prior capture, so it is safe to re-capture after a shape change).
  template <class Fn> void Capture(cudaStream_t Stream, Fn &&Body) {
    Reset();
    PLASTIX_CUDA_CHECK(
        cudaStreamBeginCapture(Stream, cudaStreamCaptureModeThreadLocal));
    Body();
    PLASTIX_CUDA_CHECK(cudaStreamEndCapture(Stream, &Graph_));
    PLASTIX_CUDA_CHECK(cudaGraphInstantiate(&Exec_, Graph_, 0));
  }

  // Replay the captured graph on Stream (async; caller synchronizes).
  void Launch(cudaStream_t Stream) const {
    PLASTIX_CUDA_CHECK(cudaGraphLaunch(Exec_, Stream));
  }

  void Reset() {
    if (Exec_) {
      cudaGraphExecDestroy(Exec_);
      Exec_ = nullptr;
    }
    if (Graph_) {
      cudaGraphDestroy(Graph_);
      Graph_ = nullptr;
    }
  }
};

} // namespace gpu
} // namespace plastix

#endif // PLASTIX_HAS_CUDA

#endif // PLASTIX_CUDA_GRAPH_HPP
