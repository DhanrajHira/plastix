add_executable(manual_fcc manual_fcc.cpp)
target_link_libraries(manual_fcc PRIVATE plastix)
target_compile_options(manual_fcc PRIVATE -Wall -Wextra -Wpedantic)
