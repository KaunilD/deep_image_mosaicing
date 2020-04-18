cmake_minimum_required(VERSION 3.10)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#OpenCV
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV library status:")
message(STATUS "    VERSION: ${OpenCV_VERSION}")
message(STATUS "    LIBS: ${OpenCV_LIBS}")
message(STATUS "    INCLUDE_PATH: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
	# Add OpenCV headers location to include paths
	include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
foreach(f ${TORCH_CXX_FLAGS})
  list(APPEND CMAKE_CUDA_FLAGS -Xcompiler ${f})
endforeach()

set(SRCS
	"extractor/superpointextractor.cpp"
	"matcher/bfmatcher.cpp"
	"imagemosaicing.cpp"
)

set(HEADERS
	
	${PROJECT_SOURCE_DIR}/include/extractor/featureextractor.hpp
	${PROJECT_SOURCE_DIR}/include/extractor/superpointextractor.hpp

	${PROJECT_SOURCE_DIR}/include/matcher/matcher.hpp
	${PROJECT_SOURCE_DIR}/include/matcher/bfmatcher.hpp

	${PROJECT_SOURCE_DIR}/include/estimator.hpp

	${PROJECT_SOURCE_DIR}/include/dtypes/descriptor.hpp
	${PROJECT_SOURCE_DIR}/include/dtypes/features.hpp
	${PROJECT_SOURCE_DIR}/include/dtypes/keypoint.hpp
	${PROJECT_SOURCE_DIR}/include/dtypes/estimatorres.hpp
	${PROJECT_SOURCE_DIR}/include/dtypes/matchpair.hpp
	
	${PROJECT_SOURCE_DIR}/include/utils.hpp
	${PROJECT_SOURCE_DIR}/include/libs.hpp
	${PROJECT_SOURCE_DIR}/include/image.hpp

	${PROJECT_SOURCE_DIR}/include/imagemosaicing.hpp
)

if(WIN32)
	add_executable("${PROJECT_NAME}"  WIN32
		${SRCS}
		${HEADERS}
	)
else()
	add_executable( "${PROJECT_NAME}" 
		${SRCS}
		${HEADERS}
	)
endif()

target_include_directories( 
	"${PROJECT_NAME}" 
		PUBLIC 
			$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:include>

)

target_link_libraries( 
	image_mosaicing "${TORCH_LIBRARIES}" ${OpenCV_LIBS}
)


set_property(TARGET image_mosaicing PROPERTY CXX_STANDARD 14)
set_target_properties(${PROJECT_NAME} PROPERTIES
	LINK_FLAGS /SUBSYSTEM:CONSOLE
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET image_mosaicing
                     POST_BUILD 
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:image_mosaicing>)
endif (MSVC)