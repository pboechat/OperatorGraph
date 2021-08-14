#include <pga/rendering/D3DBaseTriangleMesh.h>
#include <pga/rendering/D3DException.h>
#include <pga/rendering/InstancedTriangleMeshData.h>
#include <pga/rendering/RenderingConstants.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			BaseTriangleMesh::BaseTriangleMesh(size_t maxNumVertices, size_t maxNumIndices) :
				numVertices(0),
				numIndices(0),
				maxNumVertices(maxNumVertices),
				maxNumIndices(maxNumIndices),
				vertexAttributesBuffer(nullptr),
				instanceAttributesBuffer(nullptr),
				indexBuffer(nullptr)
			{
				if (this->maxNumVertices == 0)
					throw std::runtime_error("PGA::Rendering::D3D::BaseTriangleMesh::ctor(): max. num. vertices for buffer must be > 0");
				if (this->maxNumIndices == 0)
					throw std::runtime_error("PGA::Rendering::D3D::BaseTriangleMesh::ctor(): max. num. indices for buffer must be > 0");
			}

			BaseTriangleMesh::~BaseTriangleMesh()
			{
				if (vertexAttributesBuffer)
					vertexAttributesBuffer->Release();
				if (instanceAttributesBuffer)
					instanceAttributesBuffer->Release();
				if (indexBuffer)
					indexBuffer->Release();
			}

			void BaseTriangleMesh::syncVertexAttributes(ID3D11DeviceContext* deviceContext, std::vector<TriangleMeshData::VertexAttributes>& verticesAttributes, size_t numVerticesAttributes)
			{
				verticesAttributes.resize(numVerticesAttributes);
				D3D11_MAPPED_SUBRESOURCE map;
				PGA_Rendering_D3D_checkedCall(deviceContext->Map(vertexAttributesBuffer, 0, D3D11_MAP_READ, 0, &map));
				memcpy(&verticesAttributes[0], map.pData, numVerticesAttributes * sizeof(TriangleMeshData::VertexAttributes));
				deviceContext->Unmap(vertexAttributesBuffer, 0);
			}

			void BaseTriangleMesh::syncIndices(ID3D11DeviceContext* deviceContext, std::vector<unsigned int>& indices, size_t numIndices)
			{
				indices.resize(numIndices);
				D3D11_MAPPED_SUBRESOURCE map;
				PGA_Rendering_D3D_checkedCall(deviceContext->Map(indexBuffer, 0, D3D11_MAP_READ, 0, &map));
				memcpy(&indices[0], map.pData, numIndices * sizeof(unsigned int));
				deviceContext->Unmap(indexBuffer, 0);
			}

			size_t BaseTriangleMesh::getMaxNumVertices() const
			{
				return maxNumVertices;
			}

			size_t BaseTriangleMesh::getMaxNumIndices() const
			{
				return maxNumIndices;
			}

			void BaseTriangleMesh::build(ID3D11Device* device, ID3D11DeviceContext* deviceContext)
			{
				assert(sizeof(TriangleMeshData::VertexAttributes) == 48);
				D3D11_BUFFER_DESC bufferDescription;
				ZeroMemory(&bufferDescription, sizeof(D3D11_BUFFER_DESC));
				bufferDescription.Usage = D3D11_USAGE_DEFAULT;
				bufferDescription.ByteWidth = (UINT)(maxNumVertices * sizeof(TriangleMeshData::VertexAttributes));
				bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &vertexAttributesBuffer));

				bufferDescription.ByteWidth = (UINT)(maxNumIndices * sizeof(unsigned int));
				bufferDescription.BindFlags = D3D11_BIND_INDEX_BUFFER;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &indexBuffer));

				bufferDescription.Usage = D3D11_USAGE_DYNAMIC;
				bufferDescription.ByteWidth = (UINT)sizeof(InstancedTriangleMeshData::InstanceAttributes);
				bufferDescription.BindFlags = D3D11_BIND_VERTEX_BUFFER;
				bufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&bufferDescription, 0, &instanceAttributesBuffer));

				assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) % 16 == 0);

				// NOTE: writing mock per-instance data
				InstancedTriangleMeshData::InstanceAttributes mockInstanceAttributes;
				memset(&mockInstanceAttributes, 0, sizeof(InstancedTriangleMeshData::InstanceAttributes));
				mockInstanceAttributes.modelMatrix = math::identity<math::float4x4>();

				D3D11_MAPPED_SUBRESOURCE map;
				PGA_Rendering_D3D_checkedCall(deviceContext->Map(instanceAttributesBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &map));
				memcpy(map.pData, &mockInstanceAttributes, sizeof(InstancedTriangleMeshData::InstanceAttributes));
				deviceContext->Unmap(instanceAttributesBuffer, 0);
			}

			void BaseTriangleMesh::draw(ID3D11DeviceContext* deviceContext) const
			{
				if (neverDraw)
					return;
				if (numIndices == 0)
					return;
				UINT strides[2] = { sizeof(TriangleMeshData::VertexAttributes), sizeof(InstancedTriangleMeshData::InstanceAttributes) };
				UINT offsets[2] = { 0, 0 };
				ID3D11Buffer* const pBuffers[2] = { vertexAttributesBuffer, instanceAttributesBuffer };
				deviceContext->IASetVertexBuffers(0, 2, pBuffers, strides, offsets);
				deviceContext->IASetIndexBuffer(indexBuffer, DXGI_FORMAT_R32_UINT, 0);
				deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
				deviceContext->DrawIndexedInstanced((UINT)numIndices, 1, 0, 0, 0);
			}

			bool BaseTriangleMesh::hasOverflown() const
			{
				return numVertices > maxNumVertices || numIndices > maxNumIndices;
			}

			size_t BaseTriangleMesh::getNumVertices() const
			{
				return numVertices;
			}

			size_t BaseTriangleMesh::getNumIndices() const
			{
				return numIndices;
			}

		}

	}

}
