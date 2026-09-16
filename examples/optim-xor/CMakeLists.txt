add_executable(optim_xor optim_xor.cpp)
target_link_libraries(optim_xor PRIVATE plastix)
target_compile_options(optim_xor PRIVATE -Wall -Wextra -Wpedantic)
plastix_enable_cuda_on_target(optim_xor)
