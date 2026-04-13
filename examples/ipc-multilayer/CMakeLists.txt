add_executable(ipc_multilayer ipc_multilayer.cpp)
target_link_libraries(ipc_multilayer PRIVATE plastix)
target_compile_options(ipc_multilayer PRIVATE -Wall -Wextra -Wpedantic)
