#pragma once

#include "ConvexRightPrism.cuh"

#include <cuda_runtime_api.h>
#include <math/math.h>
#include <math/vector.h>

#include <cfloat>

namespace PGA
{
	namespace ContextSensitivity
	{
		struct AABB
		{
			math::float3 _min;
			math::float3 _max;

			__host__ __device__ AABB() : _min(FLT_MAX, FLT_MAX, FLT_MAX), _max(FLT_MIN, FLT_MIN, FLT_MIN) {}
			__host__ __device__ AABB(const math::float3& _min, const math::float3& _max) : _min(_min), _max(_max) {}

			__host__ __device__ __inline__ math::float3 center() const
			{
				return (_min + _max) / 2.0f;
			}

			__host__ __device__ __inline__ math::float3 size() const
			{
				return (_max - _min);
			}

			__host__ __device__ __inline__ void expand(const AABB& src)
			{
				_min = min(_min, src._min);
				_max = max(_max, src._max);
			}

			__host__ __device__ __inline__ operator Shapes::Box() const
			{
				Shapes::Box box;
				box.setPosition(center());
				box.setSize(size());
				return box;
			}

		};

	}

}
