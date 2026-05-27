#ifndef PLASTIX_MATH_HPP
#define PLASTIX_MATH_HPP
/// plastix::math - host/device-safe activation functions and their gradients.

#include "plastix/macros.hpp"

#include <cmath>

namespace plastix {
namespace math {

// ---------------------------------------------------------------------------
// Linear / identity
// ---------------------------------------------------------------------------

PLASTIX_HD float Linear(float X) { return X; }
PLASTIX_HD float LinearGrad(float /*X*/) { return 1.0f; }

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

PLASTIX_HD float ReLU(float X) { return X > 0.0f ? X : 0.0f; }

PLASTIX_HD float ReLUGradFromActivation(float A) { return A > 0.0f ? 1.0f : 0.0f; }

PLASTIX_HD float ReLUGradFromPreact(float Z) {  return Z > 0.0f ? 1.0f : 0.0f; }
// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

PLASTIX_HD float Sigmoid(float X) { return 1.0f / (1.0f + std::exp(-X)); }

PLASTIX_HD float SigmoidGradFromActivation(float A) {
  return A * (1.0f - A);
}

PLASTIX_HD float SigmoidGradFromPreact(float Z) {
  float A = Sigmoid(Z);
  return A * (1.0f - A);
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------

PLASTIX_HD float Tanh(float X) { return std::tanh(X); }

PLASTIX_HD float TanhGradFromActivation(float A) { return 1.0f - A * A; }

PLASTIX_HD float TanhGradFromPreact(float Z) {
  float A = std::tanh(Z);
  return 1.0f - A * A;
}

// ---------------------------------------------------------------------------
// GeLU (exact, error-function formulation)
// ---------------------------------------------------------------------------
//
// phi(z) = 0.5 z (1 + erf(z / sqrt(2)))
//
// d phi / d z = 0.5 (1 + erf(z / sqrt(2)))
//             + z * (1 / sqrt(2 pi)) * exp(-z^2 / 2)
//

PLASTIX_HD float GeLU(float X) {
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  return 0.5f * X * (1.0f + std::erf(X * InvSqrt2));
}

PLASTIX_HD float GeLUGradFromPreact(float Z) {
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  constexpr float InvSqrt2Pi = 0.39894228040143267794f;
  float Cdf = 0.5f * (1.0f + std::erf(Z * InvSqrt2));
  float Pdf = InvSqrt2Pi * std::exp(-0.5f * Z * Z);
  return Cdf + Z * Pdf;
}

} // namespace math
} // namespace plastix

#endif // PLASTIX_MATH_HPP
