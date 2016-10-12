cmake_minimum_required(VERSION 3.0)

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

if (PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE)
	add_definitions(-DPGA_OPERATOR_GRAPH_ANALYZER_VERBOSE)
endif()

find_package(CUDA REQUIRED)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

include_directories(
    ${CUDA_INCLUDE_DIRS}
	"${PGA_HOME}/deps/math/include"
	"${PGA_HOME}/include"
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(operator_graph_analyzer ${HEADERS} ${SOURCES})
set_target_properties(operator_graph_analyzer PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(operator_graph_analyzer pga_compiler ${CUDA_LIBRARIES})

add_custom_command(TARGET operator_graph_analyzer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/templates"
    "${PGA_BIN}/templates")
