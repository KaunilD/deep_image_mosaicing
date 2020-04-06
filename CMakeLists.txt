# CMakeList.txt : CMake project for Image Mosaicing, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

project ("image_mosaicing" 
	VERSION 1.1 
	DESCRIPTION "Image Mosaicing" 
	LANGUAGES CXX
)
# Add source to this project's executable.
add_subdirectory("src")
# TODO: Add tests and install targets if needed.
