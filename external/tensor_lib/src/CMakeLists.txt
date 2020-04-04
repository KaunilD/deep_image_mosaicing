# CMakeList.txt : CMake project for tensor_lib, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
find_package( CUDA 9.0 REQUIRED )

if(CUDA_FOUND)
	message(STATUS "CUDA library status:")
	message(STATUS "    NVCC_EXECUTABLE: ${CUDA_NVCC_EXECUTABLE}")
	message(STATUS "    INCLUDE_PATH: ${CUDA_INCLUDE_DIRS}")
	message(STATUS "    TOOLKIT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

	# compared to class settings, we let NVidia's FindCUDA CMake detect 
	# whether to build x64.  
	# Complied for 1060 architecture. Add more from here:
	# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
	set(CUDA_NVCC_FLAGS "
		-gencode;arch=compute_61,code=sm_61;"
	)
  
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
	message("CUDA is not installed on this system.")
endif()

set ( HEADERS
	${PROJECT_SOURCE_DIR}/include/device_tensor.hpp
	${PROJECT_SOURCE_DIR}/include/host_tensor.hpp
	${PROJECT_SOURCE_DIR}/include/tensor.hpp
	${PROJECT_SOURCE_DIR}/include/utils.hpp
)

set ( SRCS
	${PROJECT_SOURCE_DIR}/src/tensor.cu
	${PROJECT_SOURCE_DIR}/src/host_tensor.cu
	${PROJECT_SOURCE_DIR}/src/device_tensor.cu
)

# Add source to this project's executable.
add_library(${PROJECT_NAME} STATIC ${SRCS} ${HEADERS} )

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})


target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
target_include_directories(
	${PROJECT_NAME} 
	PUBLIC
		${CUDA_INCLUDE_DIRS}
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
)