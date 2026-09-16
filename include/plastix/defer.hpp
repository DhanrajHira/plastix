#ifndef PLASTIX_DEFER_HPP
#define PLASTIX_DEFER_HPP

// Scope guard. C++ has no `defer` keyword (cf. the C2y `defer` proposal / Go's
// defer); this is the RAII idiom. Owning types should free via their own
// destructors first; use Defer for local multi-step allocation sequences where a
// partial failure must unwind what already succeeded (dismiss() on success).

#include <utility>

namespace plastix {

template <typename F> class Defer {
  F Fn_;
  bool Active_ = true;

public:
  explicit Defer(F Fn) : Fn_(std::move(Fn)) {}
  Defer(const Defer &) = delete;
  Defer &operator=(const Defer &) = delete;
  Defer(Defer &&Other) noexcept : Fn_(std::move(Other.Fn_)), Active_(Other.Active_) {
    Other.Active_ = false;
  }
  // Cancel: the action will not run at scope exit (e.g. after success).
  void dismiss() { Active_ = false; }
  ~Defer() {
    if (Active_)
      Fn_();
  }
};

template <typename F> Defer(F) -> Defer<F>;

} // namespace plastix

#define PLASTIX_DEFER_CAT_(a, b) a##b
#define PLASTIX_DEFER_CAT(a, b) PLASTIX_DEFER_CAT_(a, b)
// PLASTIX_DEFER(stmt...): run stmt at scope exit. Capture by reference.
#define PLASTIX_DEFER(...)                                                      \
  ::plastix::Defer PLASTIX_DEFER_CAT(plastix_defer_, __LINE__) {                \
    [&]() { __VA_ARGS__; }                                                      \
  }

#endif // PLASTIX_DEFER_HPP
