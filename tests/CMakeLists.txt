add_executable(plastix_tests test_plastix.cpp test_plastix_pipeline.cpp test_alloc.cpp test_soa.cpp)
target_link_libraries(plastix_tests PRIVATE plastix GTest::gtest_main)
plastix_enable_cuda_on_target(plastix_tests)
add_test(NAME plastix_tests COMMAND plastix_tests)

# Sources must be plain ASCII (see scripts/check-ascii.sh).
add_test(NAME ascii_lint
         COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check-ascii.sh
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

if(PLASTIX_ENABLE_CUDA)
    # Lightweight unit tests for the in-house parallel primitives. Built as a
    # separate executable so the CPU build stays unchanged.
    add_executable(plastix_cuda_tests test_cuda_primitives.cpp test_device_resident.cpp)
    target_link_libraries(plastix_cuda_tests PRIVATE plastix GTest::gtest_main)
    plastix_enable_cuda_on_target(plastix_cuda_tests)
    add_test(NAME plastix_cuda_tests COMMAND plastix_cuda_tests)
endif()
