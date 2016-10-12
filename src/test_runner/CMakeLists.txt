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

set(TEST_RUNNER_SCENE "Menger Sponge" CACHE STRING "Test scene for test runner application")
set_property(CACHE TEST_RUNNER_SCENE PROPERTY STRINGS 
	"Simple House" 
	"Menger Sponge" 
	"3D Tree 12,2" 
	"3D Tree 4,3"
	"3D Tree 8,3"
	"MC Skyscrapers"
	"MC Spaceship"
	"Suburban House"
	"Balcony"
	"Commercial")

if(TEST_RUNNER_SCENE STREQUAL "Simple House")
	add_definitions(-DSIMPLE_HOUSE)
elseif(TEST_RUNNER_SCENE STREQUAL "Menger Sponge")
	add_definitions(-DMENGER_SPONGE)
elseif(TEST_RUNNER_SCENE STREQUAL "3D Tree 12,2")
	add_definitions(-D_3D_TREE_12_2)
elseif(TEST_RUNNER_SCENE STREQUAL "3D Tree 4,3")
	add_definitions(-D_3D_TREE_4_3)
elseif(TEST_RUNNER_SCENE STREQUAL "3D Tree 8,3")
	add_definitions(-D_3D_TREE_8_3)
elseif(TEST_RUNNER_SCENE STREQUAL "MC Skyscrapers")
	add_definitions(-DMC_SKYSCRAPERS)
elseif(TEST_RUNNER_SCENE STREQUAL "MC Spaceship")
	add_definitions(-DMC_SPACESHIP)
elseif(TEST_RUNNER_SCENE STREQUAL "Suburban House")
	add_definitions(-DSUBURBAN_HOUSE)
elseif(TEST_RUNNER_SCENE STREQUAL "Balcony")
	add_definitions(-DBALCONY)
elseif(TEST_RUNNER_SCENE STREQUAL "Commercial")
	add_definitions(-DCOMMERCIAL)
else()
	message(FATAL_ERROR "No test scene selected" )
endif()

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

cuda_add_executable(test_runner ${HEADERS} ${SOURCES})
set_target_properties(test_runner PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(test_runner pga_rendering whippletree ${OPENGL_gl_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_cupti_LIBRARY})

add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/models"
    "${PGA_BIN}/models")
	
add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/textures"
    "${PGA_BIN}/textures")

add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/balcony_configuration.xml"
    ${PGA_BIN})
    
add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/commercial_configuration.xml"
    ${PGA_BIN})

add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/suburban_house_configuration.xml"
    ${PGA_BIN})

add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/mc_skyscrapers_configuration.xml"
    ${PGA_BIN})

add_custom_command(TARGET test_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/mc_spaceship_configuration.xml"
    ${PGA_BIN})
