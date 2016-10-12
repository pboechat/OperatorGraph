cmake_minimum_required(VERSION 3.0)

project(zlib)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/")

file(GLOB SOURCES ${SOURCE_DIR}/*.c)

include_directories(${INCLUDE_DIR})

add_library(zlib STATIC ${SOURCES})
set_target_properties(zlib PROPERTIES DEBUG_POSTFIX "d")

set(ZLIB_INCLUDE_DIR ${INCLUDE_DIR} CACHE INTERNAL "")

