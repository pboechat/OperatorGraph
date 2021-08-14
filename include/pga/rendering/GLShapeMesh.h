#pragma once

#include "GLException.h"
#include "GLInstancedTriangleMeshSource.h"
#include "InstancedTriangleMeshData.h"
#include "RenderingConstants.h"
#include "ShapeMeshAttributes.cuh"
#include "TriangleMeshData.h"

#include <GL/glew.h>
#include <driver_types.h>
#include <math/matrix.h>
#include <math/vector.h>
#include <pga/core/ShapeGenerator.cuh>
#include <pga/core/Shapes.cuh>
#include <pga/core/TStdLib.h>
#include <windows.h>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			template <typename ShapeT>
			class ShapeMeshImpl;

			struct ShapeMeshGenerationFunction
			{
				template <typename ShapeT>
				__host__ __device__ static void perShape(ShapeT& shape, unsigned int terminalIndex, float attr1, float attr2, ShapeMeshImpl<ShapeT>* mesh)
				{
				}

				template <typename ShapeT>
				__host__ __device__ static unsigned int allocateVertices(ShapeT& shape, unsigned int terminalIndex, float attr1, float attr2, ShapeMeshImpl<ShapeT>* mesh)
				{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
					unsigned int currentSize = static_cast<unsigned int>(mesh->verticesAttributes.size());
					unsigned int newSize = currentSize + ShapeMeshAttributes<ShapeT>::getNumVertices(shape);
					mesh->verticesAttributes.resize(newSize);
					return currentSize;
#else
					return 0;
#endif
				}

				template <typename ShapeT>
				__host__ __device__ static void perVertex(unsigned int vertexIndex, const math::float4& vertex, const math::float3& normal, const math::float2& uv, unsigned int terminalIndex, float attr1, float attr2, ShapeMeshImpl<ShapeT>* mesh)
				{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
					auto& vertexAttribute = mesh->verticesAttributes[vertexIndex];
					vertexAttribute.position = vertex;
					vertexAttribute.normal = normal;
					vertexAttribute.uv = uv;
#endif
				}

				template <typename ShapeT>
				__host__ __device__ static unsigned int allocateIndices(ShapeT& shape, unsigned int terminalIndex, float attr1, float attr2, ShapeMeshImpl<ShapeT>* mesh)
				{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
					unsigned int currentSize = static_cast<unsigned int>(mesh->indices.size());
					mesh->indices.resize(currentSize + ShapeMeshAttributes<ShapeT>::getNumIndices(shape));
					return currentSize;
#else
					return 0;
#endif
				}

				template <typename ShapeT>
				__host__ __device__ static void perTriangle(unsigned int index, unsigned int vertex0Index, unsigned int vertex1Index, unsigned int vertex2Index, unsigned int terminalIndex, float attr1, float attr2, ShapeMeshImpl<ShapeT>* mesh)
				{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
					mesh->indices[index] = vertex0Index;
					mesh->indices[index + 1] = vertex1Index;
					mesh->indices[index + 2] = vertex2Index;
#endif
				}

			};

			template <typename ShapeT>
			class ShapeMeshImpl : public PGA::Rendering::GL::InstancedTriangleMeshSource
			{
			private:
				GLuint vao;
				GLuint vertexAttributesBuffer;
				GLuint indexBuffer;
				GLuint instanceAttributesBuffer;
				std::vector<TriangleMeshData::VertexAttributes> verticesAttributes;
				std::vector<InstancedTriangleMeshData::InstanceAttributes> instancesAttributes;
				std::vector<unsigned int> indices;
				bool pendingSync;

				friend struct ShapeMeshGenerationFunction;

			public:
				ShapeMeshImpl() : pendingSync(false)
				{
					ShapeT shape;
					ShapeGenerator<ShapeT, false>::template run<ShapeMeshGenerationFunction>(shape, 0, 0.0f, 0.0f, this);
				}

				ShapeMeshImpl(ShapeT& shape) : pendingSync(false)
				{
					ShapeGenerator<ShapeT, false>::template run<ShapeMeshGenerationFunction>(shape, 0, 0.0f, 0.0f, this);
				}

				ShapeMeshImpl(ShapeT&& shape) : pendingSync(false)
				{
					ShapeGenerator<ShapeT, false>::template run<ShapeMeshGenerationFunction>(shape, 0, 0.0f, 0.0f, this);
				}

				virtual ~ShapeMeshImpl()
				{
					if (vao)
						glDeleteBuffers(1, &vao);
					if (vertexAttributesBuffer)
						glDeleteBuffers(1, &vertexAttributesBuffer);
					if (instanceAttributesBuffer)
						glDeleteBuffers(1, &instanceAttributesBuffer);
					if (indexBuffer)
						glDeleteBuffers(1, &indexBuffer);
					PGA_Rendering_GL_checkError();
				}

				ShapeMeshImpl(const ShapeMeshImpl& other) = delete;
				ShapeMeshImpl& operator =(const ShapeMeshImpl& other) = delete;

				virtual size_t getNumInstances()
				{
					return instancesAttributes.size();
				}

				virtual void sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes)
				{
					this->instancesAttributes.clear();
					if (numInstances > 0)
					{
						this->instancesAttributes.resize(numInstances);
						memcpy(&this->instancesAttributes[0], instancesAttributes, numInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes));
					}
					glBindBuffer(GL_ARRAY_BUFFER, instanceAttributesBuffer);
					glBufferSubData(GL_ARRAY_BUFFER, 0, numInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes), (const void*)instancesAttributes);
					PGA_Rendering_GL_checkError();
					pendingSync = false;
				}

				virtual void sync(size_t numInstances)
				{
					if (numInstances > 0)
						instancesAttributes.resize(numInstances);
					else
						instancesAttributes.clear();
					pendingSync = true;
				}

				virtual void allocateResources(size_t maxNumInstances)
				{
					assert(sizeof(TriangleMeshData::VertexAttributes) == 48);
					assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) % 16 == 0);
					if (verticesAttributes.empty())
						throw std::runtime_error("PGA::Rendering::GL::ShapeMeshImpl::allocateResources(..): number of vertices must be greater than 0");
					if (indices.empty())
						throw std::runtime_error("PGA::Rendering::GL::ShapeMeshImpl::allocateResources(..): number of indices must be greater than 0");
					if (maxNumInstances == 0)
						throw std::runtime_error("PGA::Rendering::GL::ShapeMeshImpl::allocateResources(..): maximum number of instances must be greater than 0");
					glGenVertexArrays(1, &vao);
					glGenBuffers(1, &instanceAttributesBuffer);
					glGenBuffers(1, &vertexAttributesBuffer);
					glGenBuffers(1, &indexBuffer);
					PGA_Rendering_GL_checkError();
					size_t vertexAttributesBufferSize = verticesAttributes.size() * sizeof(TriangleMeshData::VertexAttributes);
					size_t indexBufferSize = indices.size() * sizeof(unsigned int);
					size_t instanceAttributesBufferSize = maxNumInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes);
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
					glBufferData(GL_ARRAY_BUFFER, instanceAttributesBufferSize, 0, GL_DYNAMIC_DRAW);
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
					glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferSize, &indices[0], GL_STATIC_DRAW);
					glBindVertexArray(0);
					PGA_Rendering_GL_checkError();
				}

				virtual void draw() const
				{
					if (instancesAttributes.empty())
						return;
					glBindVertexArray(vao);
					glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0, static_cast<GLsizei>(instancesAttributes.size()));
				}

				virtual GLuint getInstanceAttributesBufferRef()
				{
					return instanceAttributesBuffer;
				}

				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
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
					auto numNewVertices = this->verticesAttributes.size() * instancesAttributes.size();
					std::vector<math::float4> newPositions;
					std::vector<math::float3> newNormals;
					std::vector<math::float2> newUvs;
					newPositions.resize(numNewVertices);
					newNormals.resize(numNewVertices);
					newUvs.resize(numNewVertices);
					for (size_t i = 0, j = 0; i < instancesAttributes.size(); i++, j += this->verticesAttributes.size())
					{
						const auto& modelMatrix = transpose(instancesAttributes[i].modelMatrix);
						for (auto k = 0; k < this->verticesAttributes.size(); k++)
						{
							auto& vertexAttributes = this->verticesAttributes[k];
							auto l = j + k;
							newPositions[l] = modelMatrix * vertexAttributes.position;
							newNormals[l] = vertexAttributes.normal;
							newUvs[l] = vertexAttributes.uv;
						}
					}
					positions.insert(positions.end(), newPositions.begin(), newPositions.end());
					normals.insert(normals.end(), newNormals.begin(), newNormals.end());
					uvs.insert(uvs.end(), newUvs.begin(), newUvs.end());
					return numNewVertices;
				}

				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset /* = 0 */)
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
					unsigned int maxIndex = static_cast<unsigned int>(verticesAttributes.size());
					size_t d = 0;
					std::vector<unsigned int> newIndices;
					for (auto i = 0; i < instancesAttributes.size(); i++)
					{
						for (auto k = 0; k < this->indices.size(); k++)
						{
							auto index = this->indices[k];
							if (index >= maxIndex)
								continue;
							newIndices.emplace_back(static_cast<unsigned int>(offset + index));
							d++;
						}
						offset += verticesAttributes.size();
					}
					indices.insert(indices.end(), newIndices.begin(), newIndices.end());
					return d;
				}

			};

		}

	}

}
