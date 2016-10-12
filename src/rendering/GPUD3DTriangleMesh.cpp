#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cuda_d3d11_interop.h>

#include <math/vector.h>

#include <pga/core/CUDAException.h>
#include <pga/rendering/GPUD3DTriangleMesh.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace D3D
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
							PGA_CUDA_checkedCall(cudaGraphicsD3D11RegisterResource(&cudaVertexAttributesBuffer, vertexAttributesBuffer, cudaGraphicsRegisterFlagsNone /* the only flag working for now */));
							PGA_CUDA_checkedCall(cudaGraphicsD3D11RegisterResource(&cudaIndexBuffer, indexBuffer, cudaGraphicsRegisterFlagsNone /* the only flag working for now */));
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

				void TriangleMesh::unbind(const TriangleMeshData& data, ID3D11DeviceContext* deviceContext)
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
					std::vector<math::float2>& uvs,
					ID3D11DeviceContext* deviceContext)
				{
					size_t actualNumVertexAttributes = (numVertices > maxNumVertices) ? maxNumVertices : numVertices;
					if (actualNumVertexAttributes > 0)
					{
						std::vector<TriangleMeshData::VertexAttributes> newVertices;
						syncVertexAttributes(deviceContext, newVertices, actualNumVertexAttributes);
						for (auto& vertex : newVertices)
						{
							position.emplace_back(vertex.position);
							normals.emplace_back(vertex.normal);
							uvs.emplace_back(vertex.uv);
						}
					}
					return actualNumVertexAttributes;
				}

				size_t TriangleMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext)
				{
					size_t actualNumIndices = (numIndices > maxNumIndices) ? maxNumIndices : numIndices;
					if (actualNumIndices > 0)
					{
						size_t actualNumVertices = (numVertices > maxNumVertices) ? maxNumVertices : numVertices;
						std::vector<unsigned int> newIndices;
						syncIndices(deviceContext, newIndices, actualNumIndices);
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
