#include <GL/glew.h>
#include <pga/core/CUDAException.h>
#include <pga/rendering/CPUGLTriangleMesh.h>
#include <pga/rendering/GLException.h>
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace GL
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

				void TriangleMesh::unbind(const TriangleMeshData& data)
				{
					numVertices = static_cast<size_t>(data.numVertices);
					auto actualNumVertices = (numVertices > maxNumVertices) ? maxNumVertices : numVertices;
					if (actualNumVertices > 0)
					{
						vertices.resize(actualNumVertices);
						memcpy(&vertices[0], data.verticesAttributes, actualNumVertices * sizeof(TriangleMeshData::VertexAttributes));
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
					// NOTE: update mesh data to GL buffers
					if (actualNumVertices > 0)
					{
						glBindBuffer(GL_ARRAY_BUFFER, vertexAttributesBuffer);
						glBufferSubData(GL_ARRAY_BUFFER, 0, actualNumVertices * sizeof(TriangleMeshData::VertexAttributes), data.verticesAttributes);
						PGA_Rendering_GL_checkError();
					}
					if (actualNumIndices > 0)
					{
						glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
						glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, actualNumIndices * sizeof(unsigned int), data.indices);
						PGA_Rendering_GL_checkError();
					}
					free(data.verticesAttributes);
					free(data.indices);
				}

				size_t TriangleMesh::appendVertexAttributes(std::vector<math::float4>& positions,
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

				size_t TriangleMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset)
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
