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

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	if (CUDA_VERSION_MAJOR GREATER_EQUAL 11)
		set(CUPTI_ARCH "lib64")
		set(CUPTI_DLL "cupti64_2021.2.1.dll")
	else()
		set(CUPTI_ARCH "libx64")
		set(CUPTI_DLL "cupti64_75.dll")
	endif()
elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
	if (CUDA_VERSION_MAJOR GREATER_EQUAL 11)
		error("there's no win32 version of cupti in later versions of the CUDA SDK")
	else()
		set(CUPTI_ARCH "libx32")
		set(CUPTI_DLL "cupti32_75.dll")
	endif()
else()
	error("unknown architecture")
endif()
    
set(CUPTI_PATH "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/${CUPTI_ARCH}/${CUPTI_DLL}")
	
add_custom_command(TARGET pga_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUPTI_PATH}"
    "${PGA_BIN}")
    
add_custom_command(TARGET pga_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUPTI_PATH}"
    "${PGA_HOME}/auto_tuner/templates/bin")
