#ifndef PLASTIX_POSITION_HPP
#define PLASTIX_POSITION_HPP

#include <cstdint>

namespace plastix {

struct UnitPosition {
  _Float16 X;
  _Float16 Y;
  _Float16 Z;
  uint16_t Pad;

  bool IsZero() const {
    return X == _Float16{0} && Y == _Float16{0} && Z == _Float16{0};
  }
  explicit operator bool() const { return !IsZero(); }
};

} // namespace plastix

#endif // PLASTIX_POSITION_HPP
