cmake_minimum_required (VERSION 3.8)

message(INFO ": Building example")

add_executable(example "main.cpp")

target_link_libraries(example PRIVATE ${PROJECT_NAME})