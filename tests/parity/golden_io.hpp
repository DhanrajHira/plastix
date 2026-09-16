#ifndef PLASTIX_TESTS_PARITY_GOLDEN_IO_HPP
#define PLASTIX_TESTS_PARITY_GOLDEN_IO_HPP

// Loading and comparison helpers for the plastax conformance vectors.
//
// plastax is the reference implementation; plastix is the one under test. The
// vectors in tests/golden/ are emitted by plastax (scripts/emit_parity_goldens.py)
// and consumed here. See notes/parity/00-parity-harness.md.
//
// Comparison is tolerance-based, not exact, for everything the two
// implementations *compute*: plastix walks levels while plastax
// segment-reduces, so reduction order differs and the results are not
// bit-identical by construction. The exception is structural data -- unit
// counts, levels, edge sets, live counts -- which are integers and compared
// exactly, because a mismatch there is always a bug and never noise.
//
// Failures report the worst deviation found rather than the first, so a
// divergence can be sized ("off by 3e-3 at unit 7") instead of merely
// detected. That matters as the algorithms get more complex and the question
// shifts from "does it match" to "how far has it drifted".

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace plastix::parity {

#ifndef PLASTIX_GOLDEN_DIR
#error "PLASTIX_GOLDEN_DIR must be defined by the build (see tests/CMakeLists.txt)"
#endif

// Absolute path to a checked-in golden vector.
inline std::filesystem::path GoldenPath(const std::string &Name) {
  return std::filesystem::path{PLASTIX_GOLDEN_DIR} / Name;
}

inline nlohmann::json LoadGolden(const std::string &Name) {
  const std::filesystem::path Path = GoldenPath(Name);
  std::ifstream In(Path);
  if (!In)
    throw std::runtime_error("parity: cannot open golden vector " +
                             Path.string() +
                             " (regenerate from plastax with "
                             "`uv run python scripts/emit_parity_goldens.py`)");
  nlohmann::json J;
  In >> J;
  return J;
}

// Every *.json in tests/golden/ except the RNG pin, which has its own test and
// a different schema. Sorted so the test order is stable across filesystems.
inline std::vector<std::string> ListVectors() {
  std::vector<std::string> Names;
  for (const auto &Entry :
       std::filesystem::directory_iterator{PLASTIX_GOLDEN_DIR}) {
    if (Entry.path().extension() != ".json")
      continue;
    const std::string Name = Entry.path().filename().string();
    if (Name == "rng_uniform.json")
      continue;
    Names.push_back(Name);
  }
  std::sort(Names.begin(), Names.end());
  return Names;
}

// A mixed absolute/relative tolerance, read from a vector's "tolerance" block.
struct Tolerance {
  double Rtol = 1e-4;
  double Atol = 1e-6;

  // |a - b| <= atol + rtol * |b|, with `b` the reference (plastax) value.
  // atol is not optional: several vectors have exact-zero references (a unit
  // whose inputs cancel), and relative error alone cannot compare those.
  bool Accepts(double Got, double Want) const {
    const double Diff = std::abs(Got - Want);
    return Diff <= Atol + Rtol * std::abs(Want);
  }

  // How much of the budget a comparison used. < 1.0 passes. Reported on
  // failure so a near-miss is distinguishable from a real divergence.
  double Utilisation(double Got, double Want) const {
    const double Budget = Atol + Rtol * std::abs(Want);
    if (Budget == 0.0)
      return Got == Want ? 0.0 : std::numeric_limits<double>::infinity();
    return std::abs(Got - Want) / Budget;
  }
};

inline Tolerance ToleranceFor(const nlohmann::json &Vector,
                              const std::string &Quantity) {
  Tolerance T;
  const auto Block = Vector.find("tolerance");
  if (Block == Vector.end())
    return T;
  auto Read = [&T](const nlohmann::json &J) {
    if (auto It = J.find("rtol"); It != J.end())
      T.Rtol = It->get<double>();
    if (auto It = J.find("atol"); It != J.end())
      T.Atol = It->get<double>();
  };
  if (auto It = Block->find("default"); It != Block->end())
    Read(*It);
  if (auto It = Block->find(Quantity); It != Block->end())
    Read(*It);
  return T;
}

// Result of comparing one array of floats against its reference.
struct Deviation {
  bool Ok = true;
  size_t Count = 0;
  size_t WorstIndex = 0;
  double WorstGot = 0.0;
  double WorstWant = 0.0;
  double WorstUtilisation = 0.0;
  std::string WorstLabel;

  std::string Describe(const std::string &Quantity,
                       const Tolerance &Tol) const {
    std::ostringstream Out;
    Out << Quantity << ": worst deviation at "
        << (WorstLabel.empty() ? ("index " + std::to_string(WorstIndex))
                               : WorstLabel)
        << " -- plastix " << WorstGot << " vs plastax " << WorstWant
        << " (|diff| " << std::abs(WorstGot - WorstWant) << ", "
        << WorstUtilisation << "x the rtol=" << Tol.Rtol
        << " atol=" << Tol.Atol << " budget, over " << Count << " values)";
    return Out.str();
  }
};

// Track one comparison, keeping the worst seen so far.
inline void Observe(Deviation &Dev, const Tolerance &Tol, double Got,
                    double Want, size_t Index, const std::string &Label = {}) {
  ++Dev.Count;
  const double Used = Tol.Utilisation(Got, Want);
  if (Used > Dev.WorstUtilisation) {
    Dev.WorstUtilisation = Used;
    Dev.WorstIndex = Index;
    Dev.WorstGot = Got;
    Dev.WorstWant = Want;
    Dev.WorstLabel = Label;
  }
  if (!Tol.Accepts(Got, Want))
    Dev.Ok = false;
}

} // namespace plastix::parity

#endif // PLASTIX_TESTS_PARITY_GOLDEN_IO_HPP
