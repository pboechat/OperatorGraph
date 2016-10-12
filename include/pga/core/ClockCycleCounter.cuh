#pragma once

#ifdef _WIN32
#include <intrin.h>
#endif
#include <cuda_runtime_api.h>

namespace PGA
{
	struct ClockCycleCounter
	{
	private:
		__host__ __inline__ static uint64_t rdtsc()
		{
#ifdef _WIN32
			return __rdtsc();
#else
			unsigned int lo, hi;
			__asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
			return ((uint64_t)hi << 32) | lo;
#endif
		}

	public:
		__host__ __device__ __inline__ static uint64_t count()
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return ::clock64();
#else
			return rdtsc();
#endif
		}

	};

}
