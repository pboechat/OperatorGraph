cmake_minimum_required(VERSION 3.0)

find_package(CUDA REQUIRED)

add_definitions(-DGLEW_STATIC)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
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

if (PGA_CPU)
	add_definitions(-DPGA_CPU)
endif(PGA_CPU)

file(GLOB HEADERS 
"${PGA_HOME}/include/pga/rendering/BaseInstancedTriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/CameraParameters.h" 
"${PGA_HOME}/include/pga/rendering/ColorShader.h" 
"${PGA_HOME}/include/pga/rendering/Configuration.h" 
"${PGA_HOME}/include/pga/rendering/CPUGenerator.h" 
"${PGA_HOME}/include/pga/rendering/CPUInstancedTriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/CPUTriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/DDS.h" 
"${PGA_HOME}/include/pga/rendering/GenerationFunctions.cuh" 
"${PGA_HOME}/include/pga/rendering/Generator.h" 
"${PGA_HOME}/include/pga/rendering/GPUGenerator.h" 
"${PGA_HOME}/include/pga/rendering/GPUInstancedTriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/GPUTriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/Image.h" 
"${PGA_HOME}/include/pga/rendering/InstancedTriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/InstancedTriangleMeshData.h" 
"${PGA_HOME}/include/pga/rendering/InstancedTriangleMeshSource.h" 
"${PGA_HOME}/include/pga/rendering/LightingParameters.h" 
"${PGA_HOME}/include/pga/rendering/Material.h" 
"${PGA_HOME}/include/pga/rendering/Mesh.h" 
"${PGA_HOME}/include/pga/rendering/OBJExporter.h" 
"${PGA_HOME}/include/pga/rendering/OBJMesh.h" 
"${PGA_HOME}/include/pga/rendering/PNG.h" 
"${PGA_HOME}/include/pga/rendering/RenderingConstants.h" 
"${PGA_HOME}/include/pga/rendering/RenderingGlobalVariables.cuh" 
"${PGA_HOME}/include/pga/rendering/Shader.h" 
"${PGA_HOME}/include/pga/rendering/ShapeMesh.h" 
"${PGA_HOME}/include/pga/rendering/ShapeMeshAttributes.cuh" 
"${PGA_HOME}/include/pga/rendering/Texture.h" 
"${PGA_HOME}/include/pga/rendering/TexturedShader.h" 
"${PGA_HOME}/include/pga/rendering/tiny_obj_loader.h" 
"${PGA_HOME}/include/pga/rendering/TriangleMesh.h" 
"${PGA_HOME}/include/pga/rendering/TriangleMeshData.h" 
"${CMAKE_CURRENT_SOURCE_DIR}/BaseDDS.h" 
"${CMAKE_CURRENT_SOURCE_DIR}/DDSTextureLoader.h" 
)

file(GLOB SOURCES 
"${CMAKE_CURRENT_SOURCE_DIR}/BaseDDS.cpp" 
"${CMAKE_CURRENT_SOURCE_DIR}/Configuration.cpp" 
"${CMAKE_CURRENT_SOURCE_DIR}/DDSTextureLoader.cpp" 
"${CMAKE_CURRENT_SOURCE_DIR}/OBJExporter.cpp" 
"${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader.cc" 
)

if (PGA_D3D)
	add_definitions(-DPGA_D3D) 
	
	file(GLOB EXTRA_HEADERS 
	"${PGA_HOME}/include/pga/rendering/CPUD3DGenerator.h" 
	"${PGA_HOME}/include/pga/rendering/CPUD3DInstancedTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/CPUD3DTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/D3DBaseGenerator.h" 
	"${PGA_HOME}/include/pga/rendering/D3DBaseInstancedTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/D3DBaseTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/D3DColorShader.h" 
	"${PGA_HOME}/include/pga/rendering/D3DException.h" 
	"${PGA_HOME}/include/pga/rendering/D3DInstancedTriangleMeshSource.h" 
	"${PGA_HOME}/include/pga/rendering/D3DMaterial.h" 
	"${PGA_HOME}/include/pga/rendering/D3DMesh.h" 
	"${PGA_HOME}/include/pga/rendering/D3DOBJMesh.h" 
	"${PGA_HOME}/include/pga/rendering/D3DShader.h" 
	"${PGA_HOME}/include/pga/rendering/D3DShapeMesh.h" 
	"${PGA_HOME}/include/pga/rendering/D3DTexture.h" 
	"${PGA_HOME}/include/pga/rendering/D3DTexturedShader.h" 
	"${PGA_HOME}/include/pga/rendering/GPUD3DGenerator.h" 
	"${PGA_HOME}/include/pga/rendering/GPUD3DInstancedTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GPUD3DTriangleMesh.h" 
	"${CMAKE_CURRENT_SOURCE_DIR}/DirectXHelpers.h" 
	"${CMAKE_CURRENT_SOURCE_DIR}/dxdds.h" 
	"${CMAKE_CURRENT_SOURCE_DIR}/pch.h" 
	"${CMAKE_CURRENT_SOURCE_DIR}/PlatformHelpers.h" 
	)
	
	file(GLOB EXTRA_SOURCES 
	"${CMAKE_CURRENT_SOURCE_DIR}/CPUD3DInstancedTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/CPUD3DTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DBaseInstancedTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DBaseTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DException.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DMaterial.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DOBJMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DShader.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/D3DTexture.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GPUD3DInstancedTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GPUD3DTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp" 
	)
    
    file(GLOB SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsli ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsl)
	
else(PGA_D3D)
	file(GLOB EXTRA_HEADERS 
	"${PGA_HOME}/include/pga/rendering/CPUGLGenerator.h" 
	"${PGA_HOME}/include/pga/rendering/CPUGLInstancedTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/CPUGLTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GLBaseGenerator.h" 
	"${PGA_HOME}/include/pga/rendering/GLBaseInstancedTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GLBaseTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GLColorShader.h" 
	"${PGA_HOME}/include/pga/rendering/GLDDS.h" 
	"${PGA_HOME}/include/pga/rendering/GLException.h" 
	"${PGA_HOME}/include/pga/rendering/GLInstancedTriangleMeshSource.h" 
	"${PGA_HOME}/include/pga/rendering/GLMaterial.h" 
	"${PGA_HOME}/include/pga/rendering/GLMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GLOBJMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GLPNG.h" 
	"${PGA_HOME}/include/pga/rendering/GLShader.h" 
	"${PGA_HOME}/include/pga/rendering/GLShapeMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GLTexture.h" 
	"${PGA_HOME}/include/pga/rendering/GLTexturedShader.h" 
	"${PGA_HOME}/include/pga/rendering/GPUGLGenerator.h" 
	"${PGA_HOME}/include/pga/rendering/GPUGLInstancedTriangleMesh.h" 
	"${PGA_HOME}/include/pga/rendering/GPUGLTriangleMesh.h" 
	)
	
	file(GLOB EXTRA_SOURCES 
	"${CMAKE_CURRENT_SOURCE_DIR}/CPUGLInstancedTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/CPUGLTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLBaseInstancedTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLBaseTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLDDS.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLException.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLMaterial.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLOBJMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLPNG.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLShader.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GLTexture.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GPUGLInstancedTriangleMesh.cpp" 
	"${CMAKE_CURRENT_SOURCE_DIR}/GPUGLTriangleMesh.cpp" 
	"${PGA_HOME}/deps/glew/src/glew.c" 
	)
    
    file(GLOB SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl)

endif(PGA_D3D)

include_directories(
	${CUDA_INCLUDE_DIRS}
	${WHIPPLETREE_INCLUDE_DIR}
	${ZLIB_INCLUDE_DIR} 
	${LIBPNG_INCLUDE_DIR} 
	"${PGA_HOME}/deps/math/include"
	"${PGA_HOME}/deps/rapidxml/include"
	"${PGA_HOME}/include"
)

if (PGA_D3D)
	find_package(DirectX)
	if (NOT DirectX_D3D11_INCLUDE_FOUND)
		message(FATAL_ERROR "Missing D3D11")
	endif(NOT DirectX_D3D11_INCLUDE_FOUND)
	include_directories(${DirectX_D3D11_INCLUDE_DIR})
else(PGA_D3D)
	find_package(OpenGL REQUIRED)
	include_directories("${PGA_HOME}/deps/glew/include")
endif(PGA_D3D)

source_group("Source Files" FILES ${SOURCES} ${EXTRA_SOURCES})
source_group("Header Files" FILES ${HEADERS} ${EXTRA_HEADERS})
source_group("Shader Files" FILES ${SHADERS})

add_library(pga_rendering STATIC ${HEADERS} ${EXTRA_HEADERS} ${SOURCES} ${EXTRA_SOURCES} ${SHADERS})
set_target_properties(pga_rendering PROPERTIES DEBUG_POSTFIX "d")
target_link_libraries(pga_rendering zlib png ${CUDA_LIBRARIES})

if (PGA_D3D)
	target_link_libraries(pga_rendering ${DirectX_D3D11_LIBRARY})
else(PGA_D3D)
	target_link_libraries(pga_rendering ${OPENGL_gl_LIBRARY})
endif(PGA_D3D)

if (NOT IS_DIRECTORY "${PGA_HOME}/auto_tuner/templates/bin/shaders")
    add_custom_command(TARGET pga_rendering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory 
        "${PGA_HOME}/auto_tuner/templates/bin/shaders")
endif()

if (NOT IS_DIRECTORY "${PGA_BIN}/shaders")
    add_custom_command(TARGET pga_rendering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory 
        "${PGA_BIN}/shaders")
endif()

foreach(shader_file ${SHADERS})
    add_custom_command(TARGET pga_rendering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${shader_file}"
        "${PGA_BIN}/shaders")
        
    add_custom_command(TARGET pga_rendering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${shader_file}"
        "${PGA_HOME}/auto_tuner/templates/bin/shaders")
endforeach()
