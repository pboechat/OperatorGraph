#pragma once

#include <vector>
#include <driver_types.h>

#include <math/vector.h>

#include "TriangleMeshData.h"
#include "GLBaseTriangleMesh.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace GL
			{
				class TriangleMesh : public PGA::Rendering::GL::BaseTriangleMesh
				{
				private:
					cudaGraphicsResource_t cudaVertexAttributesBuffer;
					cudaGraphicsResource_t cudaIndexBuffer;
					bool registered;
					bool bound;

				public:
					TriangleMesh(size_t maxNumVertices, size_t maxNumIndices);
					virtual ~TriangleMesh();
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
