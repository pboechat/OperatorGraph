#include <pga/rendering/CPUD3DTriangleMesh.h>
#include <pga/rendering/D3DException.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace D3D
			{
				TriangleMesh::TriangleMesh(size_t maxNumVertices, size_t maxNumIndices) :
					BaseTriangleMesh(maxNumVertices, maxNumIndices)
				{
				}

				void TriangleMesh::bind(TriangleMeshData& data)
				{
					data.numVertices = 0;
					data.numIndices = 0;
					data.maxNumVertices = static_cast<unsigned int>(maxNumVertices);
					data.maxNumIndices = static_cast<unsigned int>(maxNumIndices);
					data.verticesAttributes = (TriangleMeshData::VertexAttributes*)malloc(maxNumVertices * sizeof(TriangleMeshData::VertexAttributes));
					data.indices = (unsigned int*)malloc(maxNumIndices * sizeof(unsigned int));
				}

				void TriangleMesh::unbind(const TriangleMeshData& data, ID3D11DeviceContext* deviceContext)
				{
					numVertices = static_cast<size_t>(data.numVertices);
					auto actualNumVertexAttributes = (numVertices > maxNumVertices) ? maxNumVertices : numVertices;
					if (actualNumVertexAttributes > 0)
					{
						vertices.resize(actualNumVertexAttributes);
						memcpy(&vertices[0], data.verticesAttributes, actualNumVertexAttributes * sizeof(TriangleMeshData::VertexAttributes));
					}
					else
						vertices.clear();
					numIndices = static_cast<size_t>(data.numIndices);
					auto actualNumIndices = (numIndices > maxNumIndices) ? maxNumIndices : numIndices;
					if (actualNumIndices > 0)
					{
						indices.resize(actualNumIndices);
						memcpy(&indices[0], data.indices, actualNumIndices * sizeof(unsigned int));
					}
					else
						indices.clear();
					D3D11_MAPPED_SUBRESOURCE map;
					if (actualNumVertexAttributes > 0)
					{
						PGA_Rendering_D3D_checkedCall(deviceContext->Map(vertexAttributesBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
						memcpy(map.pData, data.verticesAttributes, actualNumVertexAttributes * sizeof(TriangleMeshData::VertexAttributes));
						deviceContext->Unmap(vertexAttributesBuffer, 0);
					}
					if (actualNumIndices > 0)
					{
						PGA_Rendering_D3D_checkedCall(deviceContext->Map(indexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
						memcpy(map.pData, data.indices, actualNumIndices * sizeof(unsigned int));
						deviceContext->Unmap(indexBuffer, 0);
					}
					free(data.verticesAttributes);
					free(data.indices);
				}

				size_t TriangleMesh::appendVertexAttributes(ID3D11DeviceContext* deviceContext,
					std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs)
				{
					for (auto& vertex : this->vertices)
					{
						positions.emplace_back(vertex.position);
						normals.emplace_back(vertex.normal);
						uvs.emplace_back(vertex.uv);
					}
					return this->vertices.size();
				}

				size_t TriangleMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext)
				{
					std::vector<unsigned int> newIndices;
					unsigned int maxIndex = static_cast<unsigned int>(vertices.size());
					size_t d = 0;
					for (auto i = 0; i < this->indices.size(); i++)
					{
						auto index = this->indices[i];
						if (index >= maxIndex)
							continue;
						newIndices.emplace_back(static_cast<unsigned int>(offset + index));
						d++;
					}
					indices.insert(indices.end(), newIndices.begin(), newIndices.end());
					return d;
				}

				size_t TriangleMesh::getNumVertices() const
				{
					return numVertices;
				}

				size_t TriangleMesh::getNumIndices() const
				{
					return numIndices;
				}

			}

		}

	}

}
