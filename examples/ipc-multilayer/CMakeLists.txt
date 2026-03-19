set(
  example_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/ipc_multilayer.cpp"
)

add_executable(ipc_multilayer ${example_src_files})

target_link_libraries(ipc_multilayer PRIVATE plastix)
