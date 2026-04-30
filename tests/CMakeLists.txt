add_executable(plastix_tests test_plastix.cpp test_plastix_pipeline.cpp test_alloc.cpp test_soa.cpp)
target_link_libraries(plastix_tests PRIVATE plastix GTest::gtest_main)
plastix_enable_cuda_on_target(plastix_tests)
add_test(NAME plastix_tests COMMAND plastix_tests)

if(PLASTIX_ENABLE_CUDA)
    # Lightweight unit tests for the in-house parallel primitives. Built as a
    # separate executable so the CPU build stays unchanged.
    add_executable(plastix_cuda_tests test_cuda_primitives.cpp)
    target_link_libraries(plastix_cuda_tests PRIVATE plastix GTest::gtest_main)
    plastix_enable_cuda_on_target(plastix_cuda_tests)
    add_test(NAME plastix_cuda_tests COMMAND plastix_cuda_tests)
endif()
