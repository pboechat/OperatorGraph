#pragma once

#include <math/vector.h>
#include <math/matrix.h>

namespace PGA
{
	namespace Rendering
	{
#pragma pack(4)
		struct CameraParameters
		{
			math::float4x4 View;
			math::float4x4 Projection;
			math::float4x4 ViewProjection;
			math::float3 Position;

		};
#pragma pack()

	}

}
