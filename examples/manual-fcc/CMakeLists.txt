set(
  example_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/manual_fcc.cpp"
)

add_executable(manual_fcc ${example_src_files})

target_link_libraries(manual_fcc PRIVATE plastix)
