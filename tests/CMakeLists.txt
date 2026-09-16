add_executable(plastix_tests test_plastix.cpp test_plastix_pipeline.cpp test_alloc.cpp test_soa.cpp)
target_link_libraries(plastix_tests PRIVATE plastix GTest::gtest_main)
plastix_enable_cuda_on_target(plastix_tests)
add_test(NAME plastix_tests COMMAND plastix_tests)

# Sources must be plain ASCII (see scripts/check-ascii.sh).
add_test(NAME ascii_lint
         COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check-ascii.sh
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# ---------------------------------------------------------------------------
# RNG parity / golden tests (the DEEP R substrate; per-engine goldens)
# ---------------------------------------------------------------------------
#
# Kept as a separate executable from plastix_tests: it is the only test target
# with a JSON dependency. test_parity_rng_cuda.cpp guards itself on
# PLASTIX_HAS_CUDA and compiles to an empty translation unit in a CPU-only
# build.
if(PLASTIX_BUILD_PARITY_TESTS)
    add_executable(plastix_parity_tests
      test_parity_rng.cpp
      test_parity_rng_cuda.cpp)
    target_link_libraries(plastix_parity_tests
      PRIVATE plastix GTest::gtest_main nlohmann_json::nlohmann_json)
    target_include_directories(plastix_parity_tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    # Golden vectors are read from the source tree, not copied into the build
    # tree: they are checked-in fixtures, so a regenerated vector takes effect
    # without a rebuild.
    target_compile_definitions(plastix_parity_tests PRIVATE
      PLASTIX_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
    plastix_enable_cuda_on_target(plastix_parity_tests)
    add_test(NAME rng_parity COMMAND plastix_parity_tests)

    # Regenerates the per-engine RNG goldens. Not wired into the default build:
    # a golden is only ever regenerated deliberately.
    add_executable(emit_rng_golden EXCLUDE_FROM_ALL tools/emit_rng_golden.cpp)
    target_link_libraries(emit_rng_golden PRIVATE plastix)
    plastix_enable_cuda_on_target(emit_rng_golden)
endif()

if(PLASTIX_ENABLE_CUDA)
    # Lightweight unit tests for the in-house parallel primitives. Built as a
    # separate executable so the CPU build stays unchanged.
    add_executable(plastix_cuda_tests test_cuda_primitives.cpp test_device_resident.cpp)
    target_link_libraries(plastix_cuda_tests PRIVATE plastix GTest::gtest_main)
    plastix_enable_cuda_on_target(plastix_cuda_tests)
    add_test(NAME plastix_cuda_tests COMMAND plastix_cuda_tests)
endif()
