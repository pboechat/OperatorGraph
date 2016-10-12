#pragma once

#include <cassert>
#include <vector>
#include <memory>
#include <d3d11.h>
#include <stdexcept>
#include <driver_types.h>

#include <math/vector.h>
#include <math/matrix.h>

#include <pga/core/Shapes.cuh>
#include <pga/core/ShapeGenerator.cuh>
#include <pga/core/TStdLib.h>

#include "RenderingConstants.h"
#include "ShapeMeshAttributes.cuh"
#include "TriangleMeshData.h"
#include "InstancedTriangleMeshData.h"
#include "D3DInstancedTriangleMeshSource.h"
#include "D3DException.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
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
			class ShapeMeshImpl : public PGA::Rendering::D3D::InstancedTriangleMeshSource
			{
			private:
				ID3D11Buffer* vertexAttributesBuffer;
				ID3D11Buffer* indexBuffer;
				ID3D11Buffer* instanceAttributesBuffer;
				std::vector<TriangleMeshData::VertexAttributes> verticesAttributes;
				std::vector<InstancedTriangleMeshData::InstanceAttributes> instancesAttributes;
				std::vector<unsigned int> indices;
				bool pendingSync;

				void syncInstanceAttributes(ID3D11DeviceContext* deviceContext)
				{
					D3D11_MAPPED_SUBRESOURCE map;
					PGA_Rendering_D3D_checkedCall(deviceContext->Map(instanceAttributesBuffer, 0, D3D11_MAP_READ, 0, &map));
					memcpy(&instancesAttributes[0], map.pData, instancesAttributes.size() * sizeof(InstancedTriangleMeshData::InstanceAttributes));
					deviceContext->Unmap(instanceAttributesBuffer, 0);
				}

				friend struct ShapeMeshGenerationFunction;

			public:
				ShapeMeshImpl() : pendingSync(false), vertexAttributesBuffer(nullptr), indexBuffer(nullptr), instanceAttributesBuffer(nullptr)
				{
					ShapeT shape;
					ShapeGenerator<ShapeT, false>::template run<ShapeMeshGenerationFunction>(shape, 0, 0.0f, 0.0f, this);
				}

				ShapeMeshImpl(ShapeT& shape) : pendingSync(false), vertexAttributesBuffer(nullptr), indexBuffer(nullptr), instanceAttributesBuffer(nullptr)
				{
					ShapeGenerator<ShapeT, false>::template run<ShapeMeshGenerationFunction>(shape, 0, 0.0f, 0.0f, this);
				}

				ShapeMeshImpl(ShapeT&& shape) : pendingSync(false), vertexAttributesBuffer(nullptr), indexBuffer(nullptr), instanceAttributesBuffer(nullptr)
				{
					ShapeGenerator<ShapeT, false>::template run<ShapeMeshGenerationFunction>(shape, 0, 0.0f, 0.0f, this);
				}

				~ShapeMeshImpl()
				{
					if (vertexAttributesBuffer)
						vertexAttributesBuffer->Release();
					if (indexBuffer)
						indexBuffer->Release();
					if (instanceAttributesBuffer)
						instanceAttributesBuffer->Release();
				}

				ShapeMeshImpl(const ShapeMeshImpl& other) = delete;
				ShapeMeshImpl& operator =(const ShapeMeshImpl& other) = delete;

				virtual size_t getNumInstances()
				{
					return instancesAttributes.size();
				}

				virtual void sync(size_t numInstances)
				{
					if (numInstances > 0)
						instancesAttributes.resize(numInstances);
					else
						instancesAttributes.clear();
					pendingSync = true;
				}

				virtual void sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes, ID3D11DeviceContext* deviceContext)
				{
					this->instancesAttributes.clear();
					auto instanceAttributesBufferSize = numInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes);
					if (numInstances > 0)
					{
						this->instancesAttributes.resize(numInstances);
						memcpy(&this->instancesAttributes[0], instancesAttributes, instanceAttributesBufferSize);
					}
					D3D11_MAPPED_SUBRESOURCE map;
					PGA_Rendering_D3D_checkedCall(deviceContext->Map(instanceAttributesBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
					memcpy(map.pData, &this->instancesAttributes[0], instanceAttributesBufferSize);
					deviceContext->Unmap(instanceAttributesBuffer, 0);
					pendingSync = false;
				}

				virtual void allocateResources(size_t maxNumInstances, ID3D11Device* device, ID3D11DeviceContext* deviceContext)
				{
					assert(sizeof(TriangleMeshData::VertexAttributes) == 48);
					assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) % 16 == 0);
					if (verticesAttributes.empty())
						throw std::runtime_error("PGA::Rendering::D3D::ShapeMeshImpl::allocateResources(..): number of vertices must be greater than 0");
					if (indices.empty())
						throw std::runtime_error("PGA::Rendering::D3D::ShapeMeshImpl::allocateResources(..): number of indices must be greater than 0");
					if (maxNumInstances == 0)
						throw std::runtime_error("PGA::Rendering::D3D::ShapeMeshImpl::allocateResources(..): maximum number of instances must be greater than 0");
					size_t vertexAttributesBufferSize = verticesAttributes.size() * sizeof(TriangleMeshData::VertexAttributes);
					size_t indexBufferSize = indices.size() * sizeof(unsigned int);
					size_t instanceAttributesBufferSize = maxNumInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes);

					D3D11_BUFFER_DESC bufferDescription;
					ZeroMemory(&bufferDescription, sizeof(D3D11_BUFFER_DESC));
					bufferDescription.Usage = D3D11_USAGE_DYNAMIC;
					bufferDescription.ByteWidth = vertexAttributesBufferSize;
					bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
					bufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
					PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &vertexAttributesBuffer));

					bufferDescription.ByteWidth = indexBufferSize;
					bufferDescription.BindFlags = D3D10_BIND_INDEX_BUFFER;
					PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &indexBuffer));

					bufferDescription.Usage = D3D11_USAGE_DYNAMIC;
					bufferDescription.ByteWidth = instanceAttributesBufferSize;
					bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
					bufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
					PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &instanceAttributesBuffer));

					D3D11_MAPPED_SUBRESOURCE map;
					PGA_Rendering_D3D_checkedCall(deviceContext->Map(vertexAttributesBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
					memcpy(map.pData, &verticesAttributes[0], vertexAttributesBufferSize);
					deviceContext->Unmap(vertexAttributesBuffer, 0);
					PGA_Rendering_D3D_checkedCall(deviceContext->Map(indexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
					memcpy(map.pData, &indices[0], indexBufferSize);
					deviceContext->Unmap(indexBuffer, 0);
				}

				virtual void draw(ID3D11DeviceContext* deviceContext) const
				{
					if (instancesAttributes.empty())
						return;
					uint32_t strides[2] = { sizeof(TriangleMeshData::VertexAttributes), sizeof(InstancedTriangleMeshData::InstanceAttributes) };
					uint32_t offsets[2] = { 0, 0 };
					ID3D11Buffer* pBuffers[2] = { vertexAttributesBuffer, instanceAttributesBuffer };
					deviceContext->IASetVertexBuffers(0, 2, pBuffers, strides, offsets);
					deviceContext->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
					deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
					deviceContext->DrawIndexedInstanced(indices.size(), instancesAttributes.size(), 0, 0, 0);
				}

				virtual ID3D11Buffer* getInstanceAttributesBufferRef()
				{
					return instanceAttributesBuffer;
				}

				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs,
					ID3D11DeviceContext* deviceContext)
				{
					if (instancesAttributes.empty())
						return 0;
					if (pendingSync)
					{
						syncInstanceAttributes(deviceContext);
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

				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext)
				{
					if (instancesAttributes.empty())
						return 0;
					if (pendingSync)
					{
						syncInstanceAttributes(deviceContext);
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