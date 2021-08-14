#include <pga/rendering/GLColorShader.h>
#include <pga/rendering/GLException.h>
#include <pga/rendering/GLOBJMesh.h>
#include <pga/rendering/GLPNG.h>
#include <pga/rendering/GLTexture.h>
#include <pga/rendering/GLTexturedShader.h>
#include <pga/rendering/RenderingConstants.h>
#include <pga/rendering/TriangleMeshData.h>

#include <cassert>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			OBJMesh::OBJMesh(const std::string& fileName) : pendingSync(false)
			{
				std::string result = tinyobj::LoadObj(objShapes, objMaterials, fileName.c_str());
				if (!result.empty())
					throw std::runtime_error("PGA::Rendering::GL::OBJMesh::ctor(): " + result);
				for (auto& objShape : objShapes)
					subMeshes.emplace_back(new OBJShape(objShape));
			}

			OBJMesh::~OBJMesh()
			{
				if (instanceAttributesBuffer)
					glDeleteBuffers(1, &instanceAttributesBuffer);
				PGA_Rendering_GL_checkError();
				subMeshes.clear();
			}

			size_t OBJMesh::getNumInstances()
			{
				return instancesAttributes.size();
			}

			void OBJMesh::sync(size_t numInstances)
			{
				if (numInstances > 0)
					instancesAttributes.resize(numInstances);
				else
					instancesAttributes.clear();
				pendingSync = true;
			}

			void OBJMesh::sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes)
			{
				this->instancesAttributes.clear();
				if (numInstances > 0)
				{
					this->instancesAttributes.resize(numInstances);
					memcpy(&this->instancesAttributes[0], instancesAttributes, numInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes));
				}
				glBindBuffer(GL_ARRAY_BUFFER, instanceAttributesBuffer);
				glBufferSubData(GL_ARRAY_BUFFER, 0, numInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes), instancesAttributes);
				PGA_Rendering_GL_checkError();
				pendingSync = false;
			}

			void OBJMesh::allocateResources(size_t maxNumInstances)
			{
				assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) % 16 == 0);
				glGenBuffers(1, &instanceAttributesBuffer);
				size_t instanceAttributesBufferSize = maxNumInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes);
				glBindBuffer(GL_ARRAY_BUFFER, instanceAttributesBuffer);
				glBufferData(GL_ARRAY_BUFFER, instanceAttributesBufferSize, 0, GL_DYNAMIC_DRAW);
				PGA_Rendering_GL_checkError();
				for (auto& subMesh : subMeshes)
					subMesh->allocateResources(instanceAttributesBuffer, objMaterials);
			}

			GLuint OBJMesh::getInstanceAttributesBufferRef()
			{
				return instanceAttributesBuffer;
			}

			void OBJMesh::draw() const
			{
				if (instancesAttributes.empty())
					return;
				for (auto& subMesh : subMeshes)
					subMesh->draw(instancesAttributes.size());
			}

			size_t OBJMesh::appendVertexAttributes(std::vector<math::float4>& positions,
				std::vector<math::float3>& normals,
				std::vector<math::float2>& uvs)
			{
				if (instancesAttributes.empty())
					return 0;
				if (pendingSync)
				{
					glBindBuffer(GL_ARRAY_BUFFER, instanceAttributesBuffer);
					glGetBufferSubData(GL_ARRAY_BUFFER, 0, instancesAttributes.size() * sizeof(InstancedTriangleMeshData::InstanceAttributes), &instancesAttributes[0]);
					PGA_Rendering_GL_checkError();
					pendingSync = false;
				}
				size_t c = 0;
				for (auto& subMesh : subMeshes)
					c += subMesh->appendVertexAttributes(positions, normals, uvs, instancesAttributes);
				return c;
			}

			size_t OBJMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset /* = 0 */)
			{
				if (instancesAttributes.empty())
					return 0;
				if (pendingSync)
				{
					glBindBuffer(GL_ARRAY_BUFFER, instanceAttributesBuffer);
					glGetBufferSubData(GL_ARRAY_BUFFER, 0, instancesAttributes.size() * sizeof(InstancedTriangleMeshData::InstanceAttributes), &instancesAttributes[0]);
					PGA_Rendering_GL_checkError();
					pendingSync = false;
				}
				size_t c = 0;
				for (auto& subMesh : subMeshes)
					c += subMesh->appendIndices(indices, offset + c, instancesAttributes.size());
				return c;
			}

			OBJMesh::OBJShape::OBJShape(const tinyobj::shape_t& objShape) : shape(objShape)
			{
			}

			OBJMesh::OBJShape::~OBJShape()
			{
				if (vao)
					glDeleteBuffers(1, &vao);
				if (vertexAttributesBuffer)
					glDeleteBuffers(1, &vertexAttributesBuffer);
				if (indexBuffer)
					glDeleteBuffers(1, &indexBuffer);
				PGA_Rendering_GL_checkError();
			}

			void OBJMesh::OBJShape::allocateResources(GLuint instanceAttributesBuffer, const std::vector<tinyobj::material_t>& objMaterials)
			{
				assert(sizeof(TriangleMeshData::VertexAttributes) == 48);
				if ((shape.mesh.positions.size() % 3) != 0)
					throw std::runtime_error("PGA::Rendering::GL::OBJMesh::OBJShape::allocateResources(): OBJMesh doesn't support non-triangle meshes");
				auto numVertices = shape.mesh.positions.size() / 3;
				std::vector<TriangleMeshData::VertexAttributes> verticesAttributes;
				for (auto v = 0, p = 0, n = 0, t = 0; v < numVertices; v++, p += 3, n += 3, t += 2)
				{
					verticesAttributes.push_back({
						{ shape.mesh.positions[p], shape.mesh.positions[p + 1], shape.mesh.positions[p + 2], 1.0f},
						{ shape.mesh.normals[n], shape.mesh.normals[n + 1], shape.mesh.normals[n + 2] },
						{ shape.mesh.texcoords[t], shape.mesh.texcoords[t + 1] }
					});
				}
				glGenVertexArrays(1, &vao);
				glGenBuffers(1, &vertexAttributesBuffer);
				glGenBuffers(1, &indexBuffer);
				PGA_Rendering_GL_checkError();
				size_t vertexAttributesBufferSize = numVertices * sizeof(TriangleMeshData::VertexAttributes);
				size_t indexBufferSize = shape.mesh.indices.size() * sizeof(unsigned int);
				glBindVertexArray(vao);
				glBindBuffer(GL_ARRAY_BUFFER, vertexAttributesBuffer);
				glBufferData(GL_ARRAY_BUFFER, vertexAttributesBufferSize, &verticesAttributes[0], GL_STATIC_DRAW);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)0);
				size_t offset = sizeof(math::float4);
				glEnableVertexAttribArray(1);
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)offset);
				offset += sizeof(math::float3);
				glEnableVertexAttribArray(2);
				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)offset);
				glBindBuffer(GL_ARRAY_BUFFER, instanceAttributesBuffer);
				/*glEnableVertexAttribArray(3);
				glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)0);
				glVertexAttribDivisor(3, 1);
				offset += sizeof(float);*/
				offset = 0;
				assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) % sizeof(math::float4) == 0);
				auto numFloat4Chunks = sizeof(InstancedTriangleMeshData::InstanceAttributes) / sizeof(math::float4);
				for (auto i = 0u; i < numFloat4Chunks; i++)
				{
					glEnableVertexAttribArray(Constants::InstanceAttributesLocation + i);
					glVertexAttribPointer(Constants::InstanceAttributesLocation + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstancedTriangleMeshData::InstanceAttributes), (void*)offset);
					glVertexAttribDivisor(Constants::InstanceAttributesLocation + i, 1);
					offset += sizeof(math::float4);
				}
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferSize, &shape.mesh.indices[0], GL_STATIC_DRAW);
				glBindVertexArray(0);
				PGA_Rendering_GL_checkError();
			}

			void OBJMesh::OBJShape::draw(size_t numInstances) const
			{
				glBindVertexArray(vao);
				glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(shape.mesh.indices.size()), GL_UNSIGNED_INT, 0, static_cast<GLsizei>(numInstances));
				glBindVertexArray(0);
				PGA_Rendering_GL_checkError();
			}

			size_t OBJMesh::OBJShape::appendVertexAttributes(std::vector<math::float4>& positions,
				std::vector<math::float3>& normals,
				std::vector<math::float2>& uvs,
				const std::vector<InstancedTriangleMeshData::InstanceAttributes>& instancesAttributes) const
			{
				std::vector<math::float4> newPositions;
				std::vector<math::float3> newNormals;
				std::vector<math::float2> newUvs;
				auto numVertices = shape.mesh.positions.size() / 3;
				auto numNewVertices = numVertices * instancesAttributes.size();
				newPositions.resize(numNewVertices);
				newNormals.resize(numNewVertices);
				newUvs.resize(numNewVertices);
				for (size_t i = 0, j = 0; i < instancesAttributes.size(); i++, j += numVertices)
				{
					const auto& modelMatrix = transpose(instancesAttributes[i].modelMatrix);
					for (auto k = 0, l = 0, m = 0; k < numVertices; k++, l += 3, m += 2)
					{
						auto n = j + k;
						newPositions[n] = modelMatrix * math::float4(shape.mesh.positions[l], shape.mesh.positions[l + 1], shape.mesh.positions[l + 2], 1.0f);
						newNormals[n] = (modelMatrix * math::float4(shape.mesh.normals[l], shape.mesh.normals[l + 1], shape.mesh.normals[l + 2], 0.0f)).xyz();
						newUvs[n] = math::float2(shape.mesh.texcoords[m], shape.mesh.texcoords[m + 1]);
					}
				}
				positions.insert(positions.end(), newPositions.begin(), newPositions.end());
				normals.insert(normals.end(), newNormals.begin(), newNormals.end());
				uvs.insert(uvs.end(), newUvs.begin(), newUvs.end());
				return numNewVertices;
			}

			size_t OBJMesh::OBJShape::appendIndices(std::vector<unsigned int>& indices, size_t offset, size_t numInstances) const
			{
				size_t numVertices = shape.mesh.positions.size() / 3;
				unsigned int maxIndex = static_cast<unsigned int>(numVertices);
				size_t d = 0;
				std::vector<unsigned int> newIndices;
				for (auto i = 0; i < numInstances; i++)
				{
					for (auto k = 0; k < shape.mesh.indices.size(); k++)
					{
						auto index = shape.mesh.indices[k];
						if (index >= maxIndex)
							continue;
						newIndices.emplace_back(static_cast<unsigned int>(offset + index));
						d++;
					}
					offset += numVertices;
				}
				indices.insert(indices.end(), newIndices.begin(), newIndices.end());
				return d;
			}

		}

	}

}
