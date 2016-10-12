#include <cstdio>
#include <vector>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <math/vector.h>

#include <pga/core/CUDAException.h>
#include <pga/rendering/GLException.h>
#include <pga/rendering/GPUGLTriangleMesh.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace GL
			{
				TriangleMesh::TriangleMesh(size_t maxNumVertices, size_t maxNumIndices) :
					BaseTriangleMesh(maxNumVertices, maxNumIndices),
					cudaVertexAttributesBuffer(0),
					cudaIndexBuffer(0),
					bound(false),
					registered(false)
				{
				}

				TriangleMesh::~TriangleMesh()
				{
					if (registered)
					{
						if (bound)
						{
							PGA_CUDA_checkedCall(cudaGraphicsUnmapResources(1, &cudaVertexAttributesBuffer, 0));
							PGA_CUDA_checkedCall(cudaGraphicsUnmapResources(1, &cudaIndexBuffer, 0));
						}
						PGA_CUDA_checkedCall(cudaGraphicsUnregisterResource(cudaVertexAttributesBuffer));
						PGA_CUDA_checkedCall(cudaGraphicsUnregisterResource(cudaIndexBuffer));
					}
				}

				void TriangleMesh::bind(TriangleMeshData& data)
				{
					data.numVertices = 0;
					data.numIndices = 0;
					data.maxNumVertices = static_cast<unsigned int>(maxNumVertices);
					data.maxNumIndices = static_cast<unsigned int>(maxNumIndices);
					if (!bound)
					{
						if (!registered)
						{
							PGA_CUDA_checkedCall(cudaGraphicsGLRegisterBuffer(&cudaVertexAttributesBuffer, vertexAttributesBuffer, cudaGraphicsMapFlagsReadOnly /* cudaGraphicsMapFlagsWriteDiscard */));
							PGA_CUDA_checkedCall(cudaGraphicsGLRegisterBuffer(&cudaIndexBuffer, indexBuffer, cudaGraphicsMapFlagsReadOnly /* cudaGraphicsMapFlagsWriteDiscard */));
							registered = true;
						}
						PGA_CUDA_checkedCall(cudaGraphicsMapResources(1, &cudaVertexAttributesBuffer, 0));
						PGA_CUDA_checkedCall(cudaGraphicsMapResources(1, &cudaIndexBuffer, 0));
						bound = true;
					}
					size_t size;
					PGA_CUDA_checkedCall(cudaGraphicsResourceGetMappedPointer((void**)&data.verticesAttributes, &size, cudaVertexAttributesBuffer));
					PGA_CUDA_checkedCall(cudaGraphicsResourceGetMappedPointer((void**)&data.indices, &size, cudaIndexBuffer));
				}

				void TriangleMesh::unbind(const TriangleMeshData& data)
				{
					if (neverDraw)
						return;
					if (!bound)
						return;
					numVertices = static_cast<size_t>(data.numVertices);
					numIndices = static_cast<size_t>(data.numIndices);
					PGA_CUDA_checkedCall(cudaGraphicsUnmapResources(1, &cudaVertexAttributesBuffer, 0));
					PGA_CUDA_checkedCall(cudaGraphicsUnmapResources(1, &cudaIndexBuffer, 0));
					bound = false;
				}

				size_t TriangleMesh::appendVertexAttributes(std::vector<math::float4>& position,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs)
				{
					size_t actualNumVertices = (numVertices > maxNumVertices) ? maxNumVertices : numVertices;
					if (actualNumVertices > 0)
					{
						std::vector<TriangleMeshData::VertexAttributes> newVertices;
						newVertices.resize(actualNumVertices);
						glBindBuffer(GL_ARRAY_BUFFER, vertexAttributesBuffer);
						glGetBufferSubData(GL_ARRAY_BUFFER, 0, actualNumVertices * sizeof(TriangleMeshData::VertexAttributes), &newVertices[0]);
						PGA_Rendering_GL_checkError();
						for (auto& vertex : newVertices) 
						{
							position.emplace_back(vertex.position);
							normals.emplace_back(vertex.normal);
							uvs.emplace_back(vertex.uv);
						}
					}
					return actualNumVertices;
				}

				size_t TriangleMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset /* = 0 */)
				{
					size_t actualNumIndices = (numIndices > maxNumIndices) ? maxNumIndices : numIndices;
					if (actualNumIndices > 0)
					{
						size_t actualNumVertices = (numVertices > maxNumVertices) ? maxNumVertices : numVertices;
						std::vector<unsigned int> newIndices;
						newIndices.resize(actualNumIndices);
						glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
						glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, actualNumIndices * sizeof(unsigned int), &newIndices[0]);
						PGA_Rendering_GL_checkError();
						for (auto index : newIndices)
						{
							if (index >= actualNumVertices)
								continue;
							indices.emplace_back(static_cast<unsigned int>(offset + index));
						}
					}
					return actualNumIndices;
				}

			}

		}
	
	}

}
