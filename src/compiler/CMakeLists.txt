cmake_minimum_required(VERSION 3.0)

find_package(Boost 1.56 REQUIRED)
find_package(CUDA REQUIRED)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif(MSVC)

if(PGA_INVARIANT_CHECKING STREQUAL "None (0)")
	add_definitions(-DPGA_INVARIANT_CHECKING_LVL=0)
elseif(PGA_INVARIANT_CHECKING STREQUAL "Host only (1)")
	add_definitions(-DPGA_INVARIANT_CHECKING_LVL=1)
elseif(PGA_INVARIANT_CHECKING STREQUAL "Device only (2)")
	add_definitions(-DPGA_INVARIANT_CHECKING_LVL=2)
elseif(PGA_INVARIANT_CHECKING STREQUAL "Host and Device (3)")
	add_definitions(-DPGA_INVARIANT_CHECKING_LVL=2)
else()
	message(FATAL_ERROR "No invariant checking level selected" )
endif()

if (PGA_GENERATE_ANNOTATED_CODE)
	add_definitions(-DPGA_GENERATE_ANNOTATED_CODE)
endif()

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h 
    ${PGA_HOME}/include/pga/compiler/*.h
    ${PGA_HOME}/include/pga/compiler/abstraction/*.h
    ${PGA_HOME}/include/pga/compiler/analysis/*.h
    ${PGA_HOME}/include/pga/compiler/codegeneration/*.h)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(
	${Boost_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	"${PGA_HOME}/deps/math/include"
	"${PGA_HOME}/include/pga/core" 
	"${PGA_HOME}/include"
)

add_library(pga_compiler STATIC ${HEADERS} ${SOURCES})
set_target_properties(pga_compiler PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(pga_compiler ${CUDA_LIBRARIES})
