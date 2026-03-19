set(
  example_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/ipc_linear.cpp"
)

add_executable(ipc_linear ${example_src_files})

target_link_libraries(ipc_linear PRIVATE plastix)
