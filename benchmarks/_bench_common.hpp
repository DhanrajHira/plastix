// Shared infrastructure for benchmarks/ microbenches.
//
// Three concerns live here:
//   1. Output paths. Every microbench writes to build/benchmarks/_outputs/.
//      The directory is created lazily on first use.
//   2. Deterministic edge/weight draws shared across kernels.
//   3. /proc/self/status RSS reader (Linux-only).
//
// Header-only; included by each bench_*.cpp.

#ifndef PLASTIX_BENCH_COMMON_HPP
#define PLASTIX_BENCH_COMMON_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "plastix/random.hpp"

namespace bench_common {

// ---------------------------------------------------------------------------
// Output paths
// ---------------------------------------------------------------------------
//
// PLASTIX_BENCH_OUTPUT_DIR is injected via target_compile_definitions in
// benchmarks/CMakeLists.txt; the default fallback handles edge cases where a
// bench is compiled outside the project's build system.

inline std::filesystem::path OutputDir() {
#ifdef PLASTIX_BENCH_OUTPUT_DIR
  std::filesystem::path Dir = PLASTIX_BENCH_OUTPUT_DIR;
#else
  std::filesystem::path Dir = "_outputs";
#endif
  std::filesystem::create_directories(Dir);
  return Dir;
}

inline std::filesystem::path OutputFile(const std::string &Name) {
  return OutputDir() / Name;
}

// ---------------------------------------------------------------------------
// Deterministic edge predicate and weight draws.
//
// The exact same predicate and weight are used by every kernel in a given
// benchmark so the validation step can assert mathematical equivalence
// across implementations.
// ---------------------------------------------------------------------------

inline constexpr uint64_t kEdgeSeed = 0xC0FFEE5EED5ULL;
inline constexpr uint64_t kWeightSeed = 0xDEADBEEFC0DECAFEULL;
inline constexpr uint64_t kInputSeed = 0xABAD1DEAULL;

inline bool EdgeExists(size_t Src, size_t Dst, size_t Out, float Density) {
  return plastix::Bernoulli(kEdgeSeed, Src * Out + Dst, Density);
}

inline float EdgeWeight(size_t Src, size_t Dst, size_t Out) {
  return plastix::UniformReal(kWeightSeed, Src * Out + Dst, -1.0f, 1.0f);
}

inline std::vector<float> RandomVector(size_t N, uint64_t Seed = kInputSeed) {
  std::vector<float> V(N);
  for (size_t I = 0; I < N; ++I)
    V[I] = plastix::UniformReal(Seed, I, -1.0f, 1.0f);
  return V;
}

// ---------------------------------------------------------------------------
// /proc/self/status reader. Linux-only. Returns VmRSS in KB; -1 on failure.
//
// Use this from inside a bench iteration to record RSS at well-defined
// points (after construction, after first step, after K steps). The reader
// touches the kernel's procfs which performs a small allocation, so do not
// call inside the hot loop being timed.
// ---------------------------------------------------------------------------

inline long long ReadRssKb() {
  std::ifstream In("/proc/self/status");
  if (!In)
    return -1;
  std::string Line;
  while (std::getline(In, Line)) {
    if (Line.compare(0, 6, "VmRSS:") == 0) {
      long long V = -1;
      const char *S = Line.c_str() + 6;
      while (*S == ' ' || *S == '\t')
        ++S;
      V = std::strtoll(S, nullptr, 10);
      return V;
    }
  }
  return -1;
}

// Touch every physical page in a buffer so the kernel actually backs it.
// Use after constructing a sparse data structure under MAP_NORESERVE to
// distinguish "lazy-mapped capacity" from "actually committed pages."
inline void TouchPages(void *Ptr, size_t Bytes, size_t PageSize = 4096) {
  volatile char *P = static_cast<volatile char *>(Ptr);
  for (size_t I = 0; I < Bytes; I += PageSize)
    P[I] = static_cast<char>(I);
}

// ---------------------------------------------------------------------------
// CSV writer for microbench outputs. Each microbench writes a single CSV
// row per (bench-name, parameter-set) combination.
// ---------------------------------------------------------------------------

class CsvRow {
public:
  void Set(const std::string &K, const std::string &V) {
    for (size_t I = 0; I < Cols_.size(); ++I) {
      if (Cols_[I] == K) {
        Vals_[I] = V;
        return;
      }
    }
    Cols_.push_back(K);
    Vals_.push_back(V);
  }
  void Set(const std::string &K, double V) {
    char Buf[64];
    std::snprintf(Buf, sizeof(Buf), "%.6g", V);
    Set(K, std::string(Buf));
  }
  void Set(const std::string &K, long long V) { Set(K, std::to_string(V)); }
  void Set(const std::string &K, size_t V) { Set(K, std::to_string(V)); }
  void Set(const std::string &K, int V) { Set(K, std::to_string(V)); }

  const std::vector<std::string> &Cols() const { return Cols_; }
  const std::vector<std::string> &Vals() const { return Vals_; }

private:
  std::vector<std::string> Cols_;
  std::vector<std::string> Vals_;
};

// Append a row to a CSV file. Writes the header on first use if the file
// did not previously exist. Header column order is taken from the first row
// appended to a fresh file.
inline void AppendRow(const std::filesystem::path &Path, const CsvRow &Row) {
  bool Existed = std::filesystem::exists(Path);
  std::ofstream Os(Path, std::ios::app);
  if (!Os)
    return;
  if (!Existed) {
    for (size_t I = 0; I < Row.Cols().size(); ++I) {
      if (I)
        Os << ',';
      Os << Row.Cols()[I];
    }
    Os << '\n';
  }
  for (size_t I = 0; I < Row.Vals().size(); ++I) {
    if (I)
      Os << ',';
    Os << Row.Vals()[I];
  }
  Os << '\n';
}

} // namespace bench_common

#endif // PLASTIX_BENCH_COMMON_HPP
