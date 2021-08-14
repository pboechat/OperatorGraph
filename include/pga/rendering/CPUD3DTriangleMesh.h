#pragma once

#include "D3DBaseTriangleMesh.h"
#include "TriangleMeshData.h"

#include <math/vector.h>

#include <vector>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace D3D
			{
				class TriangleMesh : public PGA::Rendering::D3D::BaseTriangleMesh
				{
				private:
					std::vector<TriangleMeshData::VertexAttributes> vertices;
					std::vector<unsigned int> indices;

				public:
					TriangleMesh(size_t maxNumVertices, size_t maxNumIndices);
					virtual size_t getNumVertices() const;
					virtual size_t getNumIndices() const;
					virtual void bind(TriangleMeshData& data);
					virtual void unbind(const TriangleMeshData& data, ID3D11DeviceContext* deviceContext);
					virtual size_t appendVertexAttributes(ID3D11DeviceContext* deviceContext,
						std::vector<math::float4>& positions,
						std::vector<math::float3>& normals,
						std::vector<math::float2>& uvs);
					virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext);

				};

			}

		}

	}

}
