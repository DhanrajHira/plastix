set(
  example_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_regression.cpp"
)

add_executable(linear_regression ${example_src_files})

target_link_libraries(linear_regression PRIVATE plastix)
