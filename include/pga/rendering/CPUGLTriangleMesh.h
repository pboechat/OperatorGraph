#pragma once

#include <vector>

#include <math/vector.h>

#include "TriangleMeshData.h"
#include "GLBaseTriangleMesh.h"

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace GL
			{
				class TriangleMesh : public PGA::Rendering::GL::BaseTriangleMesh
				{
				private:
					std::vector<TriangleMeshData::VertexAttributes> vertices;
					std::vector<unsigned int> indices;

				public:
					TriangleMesh(size_t maxNumVertices, size_t maxNumIndices);
					virtual size_t getNumVertices() const;
					virtual size_t getNumIndices() const;
					virtual void bind(TriangleMeshData& data);
					virtual void unbind(const TriangleMeshData& data);
					virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
						std::vector<math::float3>& normals,
						std::vector<math::float2>& uvs);
					virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset /* = 0 */);

				};

			}

		}

	}

}
