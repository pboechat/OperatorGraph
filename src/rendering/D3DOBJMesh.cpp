#include <cassert>
#include <stdexcept>

#include <pga/rendering/RenderingConstants.h>
#include <pga/rendering/TriangleMeshData.h>
#include <pga/rendering/D3DOBJMesh.h>
#include <pga/rendering/D3DException.h>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			//////////////////////////////////////////////////////////////////////////
			OBJMesh::OBJMesh(const std::string& fileName) : pendingSync(false), instanceAttributesBuffer(nullptr)
			{
				std::string result = tinyobj::LoadObj(objShapes, objMaterials, fileName.c_str());
				if (!result.empty())
					throw std::runtime_error("PGA::Rendering::D3D::OBJMesh::ctor(): " + result);
				for (auto& objShape : objShapes)
					subMeshes.emplace_back(new OBJShape(objShape));
			}

			OBJMesh::~OBJMesh()
			{
				subMeshes.clear();
				if (instanceAttributesBuffer)
					instanceAttributesBuffer->Release();
			}

			size_t OBJMesh::getNumInstances()
			{
				return instancesAttributes.size();
			}

			void OBJMesh::syncInstanceAttributes(ID3D11DeviceContext* deviceContext)
			{
				D3D11_MAPPED_SUBRESOURCE map;
				PGA_Rendering_D3D_checkedCall(deviceContext->Map(instanceAttributesBuffer, 0, D3D11_MAP_READ, 0, &map));
				memcpy(&instancesAttributes[0], map.pData, instancesAttributes.size() * sizeof(InstancedTriangleMeshData::InstanceAttributes));
				deviceContext->Unmap(instanceAttributesBuffer, 0);
			}

			void OBJMesh::sync(size_t numInstances)
			{
				if (numInstances > 0)
					instancesAttributes.resize(numInstances);
				else
					instancesAttributes.clear();
				pendingSync = true;
			}

			void OBJMesh::sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes, ID3D11DeviceContext* deviceContext)
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

			void OBJMesh::allocateResources(size_t maxNumInstances, ID3D11Device* device, ID3D11DeviceContext* deviceContext)
			{
				assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) % 16 == 0);
				D3D11_BUFFER_DESC bufferDescription;
				ZeroMemory(&bufferDescription, sizeof(D3D11_BUFFER_DESC));
				bufferDescription.Usage = D3D11_USAGE_DEFAULT;
				bufferDescription.ByteWidth = (UINT)(maxNumInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes));
				bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &instanceAttributesBuffer));

				for (auto& subMesh : subMeshes)
					subMesh->allocateResources(instanceAttributesBuffer, objMaterials, device, deviceContext);
			}

			ID3D11Buffer* OBJMesh::getInstanceAttributesBufferRef()
			{
				return instanceAttributesBuffer;
			}

			void OBJMesh::draw(ID3D11DeviceContext* deviceContext) const
			{
				if (instancesAttributes.empty())
					return;
				for (auto& subMesh : subMeshes)
					subMesh->draw(instancesAttributes.size(), instanceAttributesBuffer, deviceContext);
			}

			size_t OBJMesh::appendVertexAttributes(std::vector<math::float4>& positions,
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
				size_t c = 0;
				for (auto& subMesh : subMeshes)
					c += subMesh->appendVertexAttributes(positions, normals, uvs, instancesAttributes);
				return c;
			}

			size_t OBJMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext)
			{
				if (instancesAttributes.empty())
					return 0;
				if (pendingSync)
				{
					syncInstanceAttributes(deviceContext);
					pendingSync = false;
				}
				size_t c = 0;
				for (auto& subMesh : subMeshes)
					c += subMesh->appendIndices(indices, offset + c, instancesAttributes.size());
				return c;
			}

			//////////////////////////////////////////////////////////////////////////
			OBJMesh::OBJShape::OBJShape(const tinyobj::shape_t& objShape) : shape(objShape), vertexAttributesBuffer(nullptr), indexBuffer(nullptr)
			{
			}

			OBJMesh::OBJShape::~OBJShape()
			{
				if (vertexAttributesBuffer)
					vertexAttributesBuffer->Release();
				if (indexBuffer)
					indexBuffer->Release();
			}

			void OBJMesh::OBJShape::allocateResources(ID3D11Buffer* instanceAttributesBuffer, const std::vector<tinyobj::material_t>& objMaterials, ID3D11Device* device, ID3D11DeviceContext* deviceContext)
			{
				assert(sizeof(TriangleMeshData::VertexAttributes) == 48);
				if ((shape.mesh.positions.size() % 3) != 0)
					throw std::runtime_error("PGA::Rendering::D3D::OBJMesh::OBJShape::allocateResources(): OBJMesh doesn't support non-triangle meshes");
				auto numVertices = shape.mesh.positions.size() / 3;
				std::vector<TriangleMeshData::VertexAttributes> verticesAttributes;
				for (auto v = 0, p = 0, n = 0, t = 0; v < numVertices; v++, p += 3, n += 3, t += 2)
				{
					verticesAttributes.push_back({
						{ shape.mesh.positions[p], shape.mesh.positions[p + 1], shape.mesh.positions[p + 2], 1.0f },
						{ shape.mesh.normals[n], shape.mesh.normals[n + 1], shape.mesh.normals[n + 2] },
						{ shape.mesh.texcoords[t], shape.mesh.texcoords[t + 1] }
					});
				}
				size_t vertexAttributesBufferSize = numVertices * sizeof(TriangleMeshData::VertexAttributes);
				size_t indexBufferSize = shape.mesh.indices.size() * sizeof(unsigned int);

				D3D11_BUFFER_DESC bufferDescription;
				ZeroMemory(&bufferDescription, sizeof(D3D11_BUFFER_DESC));
				bufferDescription.Usage = D3D11_USAGE_DYNAMIC;
				bufferDescription.ByteWidth = (UINT)vertexAttributesBufferSize;
				bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
				bufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &vertexAttributesBuffer));

				bufferDescription.ByteWidth = (UINT)indexBufferSize;
				bufferDescription.BindFlags = D3D10_BIND_INDEX_BUFFER;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &indexBuffer));

				D3D11_MAPPED_SUBRESOURCE map;
				PGA_Rendering_D3D_checkedCall(deviceContext->Map(vertexAttributesBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
				memcpy(map.pData, &verticesAttributes[0], vertexAttributesBufferSize);
				deviceContext->Unmap(vertexAttributesBuffer, 0);
				PGA_Rendering_D3D_checkedCall(deviceContext->Map(indexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
				int* indices = reinterpret_cast<int*>(map.pData);
				for (auto i = 0; i < shape.mesh.indices.size(); i += 3)
				{
#if defined(PGA_CW)
					indices[i] = shape.mesh.indices[i];
					indices[i + 1] = shape.mesh.indices[i + 1];
					indices[i + 2] = shape.mesh.indices[i + 2];
#else
					indices[i] = shape.mesh.indices[i + 2];
					indices[i + 1] = shape.mesh.indices[i + 1];
					indices[i + 2] = shape.mesh.indices[i];
#endif
				}
				deviceContext->Unmap(indexBuffer, 0);
			}

			void OBJMesh::OBJShape::draw(size_t numInstances, ID3D11Buffer* instanceAttributesBuffer, ID3D11DeviceContext* deviceContext) const
			{
				uint32_t strides[2] = { sizeof(TriangleMeshData::VertexAttributes), sizeof(InstancedTriangleMeshData::InstanceAttributes) };
				uint32_t offsets[2] = { 0, 0 };
				ID3D11Buffer* pBuffers[2] = { vertexAttributesBuffer, instanceAttributesBuffer };
				deviceContext->IASetVertexBuffers(0, 2, pBuffers, strides, offsets);
				deviceContext->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
				deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
				deviceContext->DrawIndexedInstanced((UINT)shape.mesh.indices.size(), (UINT)numInstances, 0, 0, 0);
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
