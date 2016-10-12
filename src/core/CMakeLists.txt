cmake_minimum_required(VERSION 3.0)

find_package(CUDA REQUIRED)

file(GLOB HEADERS ${PGA_HOME}/include/pga/core/*.h ${PGA_HOME}/include/pga/core/*.cuh)

include_directories(
	${WHIPPLETREE_INCLUDE_DIR}
	${CUDA_INCLUDE_DIRS}
	"${PGA_HOME}/deps/math/include"
	"${PGA_HOME}/include"
)

source_group("Header Files" FILES ${HEADERS})

add_library(pga_core STATIC ${HEADERS})
set_target_properties(pga_core PROPERTIES LINKER_LANGUAGE CXX)

if (NOT IS_DIRECTORY "${PGA_HOME}/auto_tuner/templates/bin")
    add_custom_command(TARGET pga_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory 
        "${PGA_HOME}/auto_tuner/templates/bin")
endif()

add_custom_command(TARGET pga_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/libWin32/cupti32_75.dll"
    ${PGA_BIN})

add_custom_command(TARGET pga_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/libWin32/cupti32_75.dll"
    "${PGA_HOME}/auto_tuner/templates/bin")
    
add_custom_command(TARGET pga_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/libx64/cupti64_75.dll"
    ${PGA_BIN})
    
add_custom_command(TARGET pga_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/libx64/cupti64_75.dll"
    "${PGA_HOME}/auto_tuner/templates/bin")
