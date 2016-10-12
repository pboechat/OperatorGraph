cmake_minimum_required(VERSION 3.0)

find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options /bigobj")

add_definitions(-DGLEW_STATIC)

if (PGA_CPU)
	add_definitions(-DPGA_CPU)
endif(PGA_CPU)

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

if (PGA_ENABLE_STATISTICS)
	add_definitions(-DPGA_ENABLE_STATISTICS)
endif(PGA_ENABLE_STATISTICS)

if (PGA_ENABLE_DEBUG_FLAGS)
	add_definitions(-DPGA_ENABLE_DEBUG_FLAGS)
endif(PGA_ENABLE_DEBUG_FLAGS)

option(INTERPRETER_SUPPORT_ALL_SHAPES "Add support to all shapes in interpreter application" OFF)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

source_group("Header Files" FILES ${HEADERS})
source_group("Source Files" FILES ${SOURCES})

# NOTE: GLEW has to be included before CUPTI
include_directories("${PGA_HOME}/deps/glew/include")
include_directories(
    ${CUDA_INCLUDE_DIRS}
	"${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
	${WHIPPLETREE_INCLUDE_DIR}
	"${PGA_HOME}/deps/math/include"
	"${PGA_HOME}/include"
    ${CMAKE_CURRENT_SOURCE_DIR}
)

cuda_add_executable(interpreter ${HEADERS} ${SOURCES})
set_target_properties(interpreter PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(interpreter pga_compiler pga_rendering whippletree ${OPENGL_gl_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_cupti_LIBRARY})

add_custom_command(TARGET interpreter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/interpreter_configuration.xml"
    ${PGA_BIN})
    
if (NOT IS_DIRECTORY "${PGA_BIN}/grammars")
    add_custom_command(TARGET interpreter POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/grammars"
        "${PGA_BIN}/grammars")
endif()