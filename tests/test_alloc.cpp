#include <gtest/gtest.h>

#include "plastix/alloc.hpp"

namespace {

struct IntTag {};
struct SizeTTag {};
struct BoolTag {};

using IntField = plastix::alloc::SOAField<IntTag, int>;
using SizeTField = plastix::alloc::SOAField<SizeTTag, size_t>;
using BoolField = plastix::alloc::SOAField<BoolTag, bool>;

struct TestEntity {};
using TestAllocator =
    plastix::alloc::SOAAllocator<TestEntity, IntField, SizeTField, BoolField>;

TEST(SOAAllocatorTest, AllocateAndReadBack) {
  TestAllocator Alloc(16);

  plastix::alloc::AllocId<TestEntity> Ids[5];
  for (int I = 0; I < 5; ++I) {
    Ids[I] = Alloc.Allocate();
    ASSERT_NE(Ids[I], static_cast<size_t>(-1));
    EXPECT_EQ(Ids[I], static_cast<size_t>(I));
  }

  for (int I = 0; I < 5; ++I) {
    Alloc.Get<IntTag>(Ids[I]) = I * 10;
    Alloc.Get<SizeTTag>(Ids[I]) = static_cast<size_t>(I * 100);
    Alloc.Get<BoolTag>(Ids[I]) = (I % 2 == 0);
  }

  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(Alloc.Get<IntTag>(Ids[I]), I * 10);
    EXPECT_EQ(Alloc.Get<SizeTTag>(Ids[I]), static_cast<size_t>(I * 100));
    EXPECT_EQ(Alloc.Get<BoolTag>(Ids[I]), (I % 2 == 0));
  }
}

TEST(SOAAllocatorTest, AllocateAtCapacity) {
  TestAllocator Alloc(3);

  EXPECT_NE(Alloc.Allocate(), static_cast<size_t>(-1));
  EXPECT_NE(Alloc.Allocate(), static_cast<size_t>(-1));
  EXPECT_NE(Alloc.Allocate(), static_cast<size_t>(-1));
  EXPECT_EQ(Alloc.Allocate(), static_cast<size_t>(-1));
}

} // namespace
