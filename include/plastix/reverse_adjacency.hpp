#ifndef PLASTIX_REVERSE_ADJACENCY_HPP
#define PLASTIX_REVERSE_ADJACENCY_HPP

// Reverse-adjacency (incoming-edge) index over the connection allocator.
//
// The connection allocator stores edges keyed by (FromId, ToId) — convenient
// for the per-edge forward/backward sweeps, but it gives no direct way to ask
// "which connections feed unit u?". That question is what you need to:
//   * run the forward/backward *per unit* (one thread reduces over a unit's
//     incoming edges) instead of per edge with atomicAdd into a shared
//     accumulator — which in turn lets a policy use a non-scalar Accumulator
//     (e.g. the imprinting learner's {Activation, NumConns}) on-device; and
//   * drive structural growth per unit without an O(N^2) all-pairs scan.
//
// This builds a CSR over live connections grouped by destination unit:
//   Offsets[u] .. Offsets[u+1]   index range into Incoming for unit u
//   Incoming[k]                  a connection id whose ToId == u
//   InDegree[u] = Offsets[u+1] - Offsets[u]
//
// The buffers are caller-owned (typically a managed-memory scratch allocator so
// device kernels can read them). Rebuild after any structural change; for a
// static topology it is built once. Build is O(NumConns) (counting sort).

#include "plastix/conn.hpp"
#include "plastix/macros.hpp"

#include <cstddef>
#include <cstdint>

namespace plastix {

// Build the reverse-adjacency CSR into caller-provided arrays.
//   Offsets  : length NumUnits + 1   (prefix sums; Offsets[0]=0)
//   Incoming : length >= NumLiveConns (connection ids grouped by ToId)
//   WritePos : scratch, length NumUnits (cursor per unit; clobbered)
// Dead (tombstoned) connections are skipped. Returns the number of live edges
// written into Incoming. Host function; the arrays it fills can live in managed
// memory for device consumption.
template <typename CA>
inline std::size_t BuildReverseAdjacency(const CA &ConnAlloc, std::size_t NumUnits,
                                         std::uint32_t *Offsets,
                                         std::uint32_t *Incoming,
                                         std::uint32_t *WritePos) {
  const std::size_t NumConns = ConnAlloc.Size();
  for (std::size_t U = 0; U <= NumUnits; ++U)
    Offsets[U] = 0;

  // Pass 1: histogram of in-degree per destination unit.
  for (std::size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto To = GetField<ToIdTag>(ConnAlloc, C);
    ++Offsets[static_cast<std::size_t>(To) + 1];
  }

  // Prefix sum -> bucket offsets.
  std::uint32_t Sum = 0;
  for (std::size_t U = 0; U < NumUnits; ++U) {
    Sum += Offsets[U + 1];
    Offsets[U + 1] = Sum;
    WritePos[U] = Offsets[U];
  }

  // Pass 2: scatter connection ids into their destination's bucket.
  for (std::size_t C = 0; C < NumConns; ++C) {
    if (GetField<DeadTag>(ConnAlloc, C))
      continue;
    auto To = static_cast<std::size_t>(GetField<ToIdTag>(ConnAlloc, C));
    Incoming[WritePos[To]++] = static_cast<std::uint32_t>(C);
  }
  return static_cast<std::size_t>(Sum);
}

// Lightweight device-readable view: pointers + count. Cheap to pass by value
// into a kernel (mirrors the StepView / allocator-by-value convention).
struct ReverseAdjacencyView {
  const std::uint32_t *Offsets = nullptr; // length NumUnits + 1
  const std::uint32_t *Incoming = nullptr;
  std::size_t NumUnits = 0;

  PLASTIX_HD std::uint32_t Begin(std::size_t U) const { return Offsets[U]; }
  PLASTIX_HD std::uint32_t End(std::size_t U) const { return Offsets[U + 1]; }
  PLASTIX_HD std::uint32_t InDegree(std::size_t U) const {
    return Offsets[U + 1] - Offsets[U];
  }
  PLASTIX_HD std::uint32_t Conn(std::uint32_t K) const { return Incoming[K]; }
};

} // namespace plastix

#endif // PLASTIX_REVERSE_ADJACENCY_HPP
