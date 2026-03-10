#include <gtest/gtest.h>

#include "plastix/plastix.hpp"

TEST(PlastixTest, Version) {
  EXPECT_STREQ(plastix::version(), "0.1.0");
}
