#pragma once

#include "D3DMesh.h"
#include "TriangleMeshData.h"

#include <d3d11.h>
#include <math/vector.h>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class BaseTriangleMesh : public PGA::Rendering::D3D::Mesh
			{
			protected:
				ID3D11Buffer* vertexAttributesBuffer;
				ID3D11Buffer* instanceAttributesBuffer;
				ID3D11Buffer* indexBuffer;
				size_t numVertices;
				size_t numIndices;
				size_t maxNumVertices;
				size_t maxNumIndices;

				BaseTriangleMesh(size_t maxNumVertices, size_t maxNumIndices);
				virtual ~BaseTriangleMesh();

				void syncVertexAttributes(ID3D11DeviceContext* deviceContext, std::vector<TriangleMeshData::VertexAttributes>& verticesAttributes, size_t numVerticesAttributes);
				void syncIndices(ID3D11DeviceContext* deviceContext, std::vector<unsigned int>& indices, size_t numIndices);

			public:
				BaseTriangleMesh(BaseTriangleMesh&) = delete;
				BaseTriangleMesh& operator=(BaseTriangleMesh&) = delete;

				bool hasOverflown() const;
				size_t getNumVertices() const;
				size_t getNumIndices() const;
				size_t getMaxNumVertices() const;
				size_t getMaxNumIndices() const;
				virtual void draw(ID3D11DeviceContext* deviceContext) const;
				void build(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
				virtual void bind(TriangleMeshData& data) = 0;
				virtual void unbind(const TriangleMeshData& data, ID3D11DeviceContext* deviceConcext) = 0;

			};

		}

	}

}