set(
  example_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/imprinting_learner.cpp"
)

add_executable(imprinting_learner ${example_src_files})

target_link_libraries(imprinting_learner PRIVATE plastix)

if(PLASTIX_BUILD_TESTS)
  add_executable(imprinting_learner_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/test_imprinting_learner.cpp"
  )
  target_link_libraries(imprinting_learner_tests PRIVATE plastix GTest::gtest_main)
  target_include_directories(imprinting_learner_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  add_test(NAME imprinting_learner_tests COMMAND imprinting_learner_tests)
endif()
