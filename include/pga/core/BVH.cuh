#pragma once

#include <cuda_runtime_api.h>

namespace PGA
{
	namespace ContextSensitivity
	{
		typedef unsigned long long BVHMask;

		template <typename BoundingVolumeT>
		struct BVH
		{
			BVHMask mask;
			BoundingVolumeT boundingVolume;

			__host__ __device__ BVH() : mask(0) {}
			__host__ __device__ BVH(BVHMask mask, const BoundingVolumeT& boundingVolume) : mask(mask), boundingVolume(boundingVolume) {}

			__host__ __device__ __inline__ void propagate(const BVH<BoundingVolumeT>& other)
			{
				mask |= other.mask;
				boundingVolume.expand(other.boundingVolume);
			}

		};

	}

}
