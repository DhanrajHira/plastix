set(
  example_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/fcc_nn.cpp"
)

add_executable(fcc_nn ${example_src_files})

target_link_libraries(fcc_nn PRIVATE plastix)
