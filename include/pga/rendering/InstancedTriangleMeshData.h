#pragma once

#include <math/matrix.h>

namespace PGA
{
	namespace Rendering
	{
		struct InstancedTriangleMeshData
		{
			__declspec(align(16))
			struct InstanceAttributes
			{
				math::float4x4 modelMatrix;
				float custom;

			};

			unsigned int numInstances;
			unsigned int maxNumInstances;
			InstanceAttributes* instancesAttributes;

			InstancedTriangleMeshData() = default;
			InstancedTriangleMeshData(const InstancedTriangleMeshData&) = delete;

		};

	}

}
