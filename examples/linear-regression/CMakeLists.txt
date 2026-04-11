add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression PRIVATE plastix)
target_compile_options(linear_regression PRIVATE -Wall -Wextra -Wpedantic)
