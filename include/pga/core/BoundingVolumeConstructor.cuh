#pragma once

#include "AABB.cuh"
#include "Shapes.cuh"

#include <cuda_runtime_api.h>
#include <math/matrix.h>
#include <math/vector.h>

#include <cfloat>

namespace PGA
{
	namespace ContextSensitivity
	{
		template <typename BoundingVolumeT>
		struct BoundingVolumeConstructor;

		template <>
		struct BoundingVolumeConstructor<AABB>
		{
			__host__ __device__ __inline__ static AABB construct(const Shapes::Box& box)
			{
				math::float3 _min(FLT_MAX, FLT_MAX, FLT_MAX), _max(FLT_MIN, FLT_MIN, FLT_MIN);
				math::float3 halfExtents = box.getHalfExtents();
				const math::float3x4& model = box.getModel();

				math::float3 point = (model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);

				return AABB(_min, _max);
			}

			__host__ __device__ __inline__ static AABB construct(const Shapes::Quad& quad)
			{
				math::float3 _min(FLT_MAX, FLT_MAX, FLT_MAX), _max(FLT_MIN, FLT_MIN, FLT_MIN);
				math::float3 halfExtents = quad.getSize() * 0.5f;
				const math::float3x4& model = quad.getModel();

				math::float3 point = (model * math::float4(-halfExtents.x, halfExtents.y, 0.0f, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(-halfExtents.x, -halfExtents.y, 0.0f, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(halfExtents.x, -halfExtents.y, 0.0f, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);
				point = (model * math::float4(halfExtents.x, halfExtents.y, 0.0f, 1.0f)).xyz();
				_min = min(_min, point);
				_max = max(_max, point);

				return AABB(_min, _max);
			}

			__host__ __device__ __inline__ static AABB construct(const Shapes::Sphere& sphere)
			{
				float halfRadius = sphere.getRadius() * 0.5f;
				math::float4x4 transform = math::float4x4::translate(sphere.getPosition());
				return AABB((transform * math::float4(-halfRadius, -halfRadius, -halfRadius, 1.0f)).xyz(), (transform * math::float4(halfRadius, halfRadius, halfRadius, 1.0f)).xyz());
			}

		};

	}

}
