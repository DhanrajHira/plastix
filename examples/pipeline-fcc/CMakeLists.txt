add_executable(pipeline_fcc pipeline_fcc.cpp)
target_link_libraries(pipeline_fcc PRIVATE plastix)
target_compile_options(pipeline_fcc PRIVATE -Wall -Wextra -Wpedantic)
