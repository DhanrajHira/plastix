add_executable(fcc_nn fcc_nn.cpp)
target_link_libraries(fcc_nn PRIVATE plastix)
target_compile_options(fcc_nn PRIVATE -Wall -Wextra -Wpedantic)
