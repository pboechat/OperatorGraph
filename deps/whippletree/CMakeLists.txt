cmake_minimum_required(VERSION 3.0)

project(whippletree)

find_package(CUDA REQUIRED)

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

file(GLOB HEADERS ${INCLUDE_DIR}/*.h ${INCLUDE_DIR}/*.cuh)
file(GLOB SOURCES ${SOURCE_DIR}/*.cpp)

include_directories(
	${CUDA_INCLUDE_DIRS}
	${INCLUDE_DIR}
)

add_library(whippletree STATIC ${HEADERS} ${SOURCES})
set_target_properties(whippletree PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(whippletree ${CUDA_LIBRARIES})

set(WHIPPLETREE_INCLUDE_DIR ${INCLUDE_DIR} CACHE INTERNAL "")

