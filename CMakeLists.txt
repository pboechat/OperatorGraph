cmake_minimum_required(VERSION 3.0)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif(CMAKE_COMPILER_IS_GNUCC)

set(PGA_INVARIANT_CHECKING "None (0)" CACHE STRING "Invariant checking for host/device code in PGA")
set_property(CACHE PGA_INVARIANT_CHECKING PROPERTY STRINGS 
	"None (0)" 
	"Host only (1)" 
	"Device only (2)" 
	"Host and Device (3)")
	
option(PGA_CPU "Use CPU for grammar evaluation (default=GPU)" OFF)
option(PGA_CW "Generate geometry in clockwise order (default=counter-clockwise)" OFF)
option(PGA_D3D "Use D3D instead of OpenGL (default=OpenGL)" OFF)
option(PGA_ENABLE_STATISTICS "Enable statistics output for all applications" ON)
option(PGA_ENABLE_DEBUG_FLAGS "Enable verbose output for all applications" OFF)
option(PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE "Enable verbose output for PGA operator graph analyzer tool" OFF)
option(PGA_GENERATE_ANNOTATED_CODE "Generate code annotations in PGA" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(PGA_HOME ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(PGA_BIN ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(PGA_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/libs)

mark_as_advanced(CMAKE_INSTALL_PREFIX)

project(pga)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PGA_BIN})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PGA_LIBRARIES})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PGA_LIBRARIES})
foreach(CMAKE_CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CMAKE_CONFIGURATION_TYPE} CMAKE_CONFIGURATION_TYPE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CMAKE_CONFIGURATION_TYPE} ${PGA_BIN})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CMAKE_CONFIGURATION_TYPE} ${PGA_LIBRARIES})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_CONFIGURATION_TYPE} ${PGA_LIBRARIES})
endforeach(CMAKE_CONFIGURATION_TYPE CMAKE_CONFIGURATION_TYPES)

add_subdirectory(deps)
add_subdirectory(src)
