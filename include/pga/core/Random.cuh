#pragma once

#include <climits>
#include <cuda_runtime_api.h>

#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "CUDAException.h"

namespace PGA
{
	namespace Random
	{
		__host__ __device__ __inline__ unsigned int tausStep(unsigned int &z, int S1, int S2, int S3, unsigned int M)
		{
			unsigned b = (((z << S1) ^ z) >> S2);
			return z = (((z & M) << S3) ^ b);
		}

		__host__ __device__ __inline__ unsigned int lcgStep(unsigned int &z, unsigned int A, unsigned int C)
		{
			return z = (A * z + C);
		}

		__host__ __device__ __inline__ float hybridTaus(float base, unsigned int modifier)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			auto z = (__float2uint_rd(base) * UINT_MAX) + (modifier * 41231278) % UINT_MAX;
#else
			auto z = (static_cast<unsigned int>(base) * UINT_MAX) + (modifier * 41231278) % UINT_MAX;
#endif
			unsigned int z1 = (z + 7873234) % UINT_MAX;
			unsigned int z2 = (z1 + 1237217827) % UINT_MAX;
			unsigned int z3 = (z2 + 87382) % UINT_MAX;
			unsigned int z4 = (z3 + 1410022) % UINT_MAX;
			// Combined period is lcm(p1,p2,p3,p4)~ 2^121  
			return (float)(2.3283064365387e-10 * (        // Periods  
				tausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1  
				tausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1  
				tausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1  
				lcgStep(z4, 1664525, 1013904223UL)        // p4=2^32  
			));
		}

		__host__ __device__ __inline__ static float nextSeed(float base, unsigned int modifier)
		{
			return hybridTaus(base, modifier);
		}

		__host__ __device__ __inline__ static float range(float seed, float min, float max)
		{
			if (seed < 1.0f)
				return seed * (max - min) + min;
			else
				return max;
		}

	}

}
