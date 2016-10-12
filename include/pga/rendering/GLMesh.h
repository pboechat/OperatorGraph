#pragma once

#include <vector>

#include <math/vector.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
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

				virtual void draw() const = 0;
				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs) = 0;
				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset = 0) = 0;

			};

		}

	}

}
