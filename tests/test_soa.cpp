#include <gtest/gtest.h>

#define PLASTIX_SOA_MODE_TAGS
#include "plastix/soa.hpp"
#include "test_soa_fields.inc"

#define PLASTIX_SOA_MODE_ALLOC
#include "plastix/soa.hpp"
#include "test_soa_fields.inc"

#define PLASTIX_SOA_MODE_HANDLE
#include "plastix/soa.hpp"
#include "test_soa_fields.inc"

namespace {

TEST(SOAMacroTest, AllocateAndReadWrite) {
  TestSOAAllocator Alloc(16);

  TestSOAId Id = Alloc.Allocate();
  ASSERT_NE(Id, static_cast<size_t>(-1));

  Alloc.Get<PosXTag>(Id) = 1.0f;
  Alloc.Get<PosYTag>(Id) = 2.0f;
  Alloc.Get<MassTag>(Id) = 3.0;

  EXPECT_FLOAT_EQ(Alloc.Get<PosXTag>(Id), 1.0f);
  EXPECT_FLOAT_EQ(Alloc.Get<PosYTag>(Id), 2.0f);
  EXPECT_DOUBLE_EQ(Alloc.Get<MassTag>(Id), 3.0);
}

TEST(SOAMacroTest, HandleAccessors) {
  TestSOAAllocator Alloc(16);

  TestSOAHandle H = Alloc.Allocate();

  H.GetPosX(Alloc) = 10.0f;
  H.GetPosY(Alloc) = 20.0f;
  H.GetMass(Alloc) = 30.0;

  EXPECT_FLOAT_EQ(H.GetPosX(Alloc), 10.0f);
  EXPECT_FLOAT_EQ(H.GetPosY(Alloc), 20.0f);
  EXPECT_DOUBLE_EQ(H.GetMass(Alloc), 30.0);
}

TEST(SOAMacroTest, HandleConversion) {
  TestSOAAllocator Alloc(16);

  TestSOAId Id = Alloc.Allocate();
  TestSOAHandle H(Id);

  // Implicit conversion back to AllocId
  TestSOAId Recovered = H;
  EXPECT_EQ(Id, Recovered);

  // Implicit construction from AllocId
  TestSOAHandle H2 = Alloc.Allocate();
  Alloc.Get<PosXTag>(H2) = 42.0f;
  EXPECT_FLOAT_EQ(H2.GetPosX(Alloc), 42.0f);
}

TEST(SOAMacroTest, CapacityExhaustion) {
  TestSOAAllocator Alloc(3);

  EXPECT_NE(Alloc.Allocate(), static_cast<size_t>(-1));
  EXPECT_NE(Alloc.Allocate(), static_cast<size_t>(-1));
  EXPECT_NE(Alloc.Allocate(), static_cast<size_t>(-1));
  EXPECT_EQ(Alloc.Allocate(), static_cast<size_t>(-1));
}

} // namespace

namespace my_ns {

#define PLASTIX_SOA_MODE_TAGS
#include "plastix/soa.hpp"

SOA_TYPE(NSEntity)
FIELD(Value, int)
SOA_END()

#define PLASTIX_SOA_MODE_ALLOC
#include "plastix/soa.hpp"

SOA_TYPE(NSEntity)
FIELD(Value, int)
SOA_END()

#define PLASTIX_SOA_MODE_HANDLE
#include "plastix/soa.hpp"

SOA_TYPE(NSEntity)
FIELD(Value, int)
SOA_END()

TEST(SOAMacroTest, NamespaceUsage) {
  NSEntityAllocator Alloc(8);
  NSEntityHandle H = Alloc.Allocate();
  H.GetValue(Alloc) = 99;
  EXPECT_EQ(H.GetValue(Alloc), 99);
}

} // namespace my_ns
