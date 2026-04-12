add_executable(imprinting_learner imprinting_learner.cpp)
target_link_libraries(imprinting_learner PRIVATE plastix)
target_compile_options(imprinting_learner PRIVATE -Wall -Wextra -Wpedantic)
