# CMakeList.txt : CMake project for tensor_lib, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("tensor_lib" LANGUAGES CUDA CXX)

add_subdirectory("src")
add_subdirectory("examples")
