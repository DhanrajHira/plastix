add_executable(plastix_tests test_plastix.cpp)
target_link_libraries(plastix_tests PRIVATE plastix GTest::gtest_main)
add_test(NAME plastix_tests COMMAND plastix_tests)
