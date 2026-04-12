add_executable(swifttd swifttd.cpp)
target_link_libraries(swifttd PRIVATE plastix)
target_compile_options(swifttd PRIVATE -Wall -Wextra -Wpedantic)
