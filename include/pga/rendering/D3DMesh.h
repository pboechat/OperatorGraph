#pragma once

#include <d3d11.h>
#include <math/vector.h>

#include <vector>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class Mesh
			{
			protected:
				bool neverDraw;

			public:
				Mesh() : neverDraw(false) {}

				virtual void setNeverDraw(bool neverDraw)
				{
					this->neverDraw = neverDraw;
				}

				virtual void draw(ID3D11DeviceContext* deviceContext) const = 0;
				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs,
					ID3D11DeviceContext* deviceContext) = 0;
				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext) = 0;

			};

		}

	}

}
