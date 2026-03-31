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

TEST(SOAAllocatorTest, GatherReordersAllFields) {
  TestAllocator Alloc(8);

  for (int I = 0; I < 4; ++I) {
    auto Id = Alloc.Allocate();
    Alloc.Get<IntTag>(Id) = I * 10;        // 0, 10, 20, 30
    Alloc.Get<SizeTTag>(Id) = I * 100u;    // 0, 100, 200, 300
    Alloc.Get<BoolTag>(Id) = (I % 2 == 0); // T, F, T, F
  }

  // Reverse order permutation: [3, 2, 1, 0]
  size_t *Perm = Alloc.PermutationScratch();
  Perm[0] = 3;
  Perm[1] = 2;
  Perm[2] = 1;
  Perm[3] = 0;
  Alloc.Gather(4);

  EXPECT_EQ(Alloc.Get<IntTag>(0), 30);
  EXPECT_EQ(Alloc.Get<IntTag>(1), 20);
  EXPECT_EQ(Alloc.Get<IntTag>(2), 10);
  EXPECT_EQ(Alloc.Get<IntTag>(3), 0);

  EXPECT_EQ(Alloc.Get<SizeTTag>(0), 300u);
  EXPECT_EQ(Alloc.Get<SizeTTag>(1), 200u);
  EXPECT_EQ(Alloc.Get<SizeTTag>(2), 100u);
  EXPECT_EQ(Alloc.Get<SizeTTag>(3), 0u);

  EXPECT_EQ(Alloc.Get<BoolTag>(0), false);
  EXPECT_EQ(Alloc.Get<BoolTag>(1), true);
  EXPECT_EQ(Alloc.Get<BoolTag>(2), false);
  EXPECT_EQ(Alloc.Get<BoolTag>(3), true);
}

} // namespace
