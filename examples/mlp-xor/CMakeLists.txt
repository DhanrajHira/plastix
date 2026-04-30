add_executable(mlp_xor mlp_xor.cpp)
target_link_libraries(mlp_xor PRIVATE plastix)
target_compile_options(mlp_xor PRIVATE -Wall -Wextra -Wpedantic)
plastix_enable_cuda_on_target(mlp_xor)
