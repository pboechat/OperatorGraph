#pragma once

#include <math/vector.h>

namespace PGA
{
	namespace Rendering
	{
		struct TriangleMeshData
		{
			__declspec(align(16))
			struct VertexAttributes
			{
				math::float4 position;
				math::float3 normal;
				math::float2 uv;

			};

			unsigned int numVertices;
			unsigned int numIndices;
			unsigned int maxNumVertices;
			unsigned int maxNumIndices;
			VertexAttributes* verticesAttributes;
			unsigned int* indices;

			TriangleMeshData() = default;
			TriangleMeshData(const TriangleMeshData&) = delete;

		};

	}

}
