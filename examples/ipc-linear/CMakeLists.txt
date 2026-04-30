add_executable(ipc_linear ipc_linear.cpp)
target_link_libraries(ipc_linear PRIVATE plastix)
target_compile_options(ipc_linear PRIVATE -Wall -Wextra -Wpedantic)
plastix_enable_cuda_on_target(ipc_linear)
