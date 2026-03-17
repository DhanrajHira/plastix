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

// ---------------------------------------------------------------------------
// PageAllocator tests
// ---------------------------------------------------------------------------

constexpr size_t TestSlotSize = 4;

struct TestPage {
  std::array<float, TestSlotSize> Data;
  const float &GetSlot(size_t I) const { return Data[I]; }
  float &WriteSlot(size_t I) { return Data[I]; }
};

struct NotAPage {
  float X;
};

// PageType concept checks
static_assert(plastix::alloc::PageType<TestPage>);
static_assert(
    plastix::alloc::PageType<plastix::alloc::Page<float, TestSlotSize>>);
static_assert(!plastix::alloc::PageType<NotAPage>);
static_assert(!plastix::alloc::PageType<float>);

struct PageEntity {};
struct TestPageTag {};
struct MetaTag {};

using TestPageAlloc = plastix::alloc::PageAllocator<
    PageEntity, plastix::alloc::SOAField<TestPageTag, TestPage>,
    plastix::alloc::SOAField<MetaTag, plastix::alloc::Page<int, TestSlotSize>>>;

TEST(PageAllocatorTest, AllocateAndAccessSlots) {
  TestPageAlloc Alloc(16);

  auto Id = Alloc.Allocate();
  ASSERT_NE(Id, static_cast<size_t>(-1));

  auto &Page = Alloc.Get<TestPageTag>(Id);
  for (size_t I = 0; I < TestSlotSize; ++I)
    Page.WriteSlot(I) = static_cast<float>(I) * 1.5f;

  for (size_t I = 0; I < TestSlotSize; ++I)
    EXPECT_FLOAT_EQ(Page.GetSlot(I), static_cast<float>(I) * 1.5f);
}

TEST(PageAllocatorTest, ParallelMetadataField) {
  TestPageAlloc Alloc(16);

  auto Id = Alloc.Allocate();
  ASSERT_NE(Id, static_cast<size_t>(-1));

  auto &Meta = Alloc.Get<MetaTag>(Id);
  for (size_t I = 0; I < TestSlotSize; ++I)
    Meta.WriteSlot(I) = static_cast<int>(I) * 10;

  for (size_t I = 0; I < TestSlotSize; ++I)
    EXPECT_EQ(Meta.GetSlot(I), static_cast<int>(I) * 10);
}

TEST(PageAllocatorTest, MultiplePages) {
  TestPageAlloc Alloc(16);

  auto Id0 = Alloc.Allocate();
  auto Id1 = Alloc.Allocate();
  ASSERT_NE(Id0, Id1);

  Alloc.Get<TestPageTag>(Id0).WriteSlot(0) = 42.0f;
  Alloc.Get<TestPageTag>(Id1).WriteSlot(0) = 99.0f;

  EXPECT_FLOAT_EQ(Alloc.Get<TestPageTag>(Id0).GetSlot(0), 42.0f);
  EXPECT_FLOAT_EQ(Alloc.Get<TestPageTag>(Id1).GetSlot(0), 99.0f);
}

} // namespace
