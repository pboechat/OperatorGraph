#pragma once

#include <string.h>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <math/vector.h>

#include "GlobalConstants.h"
#include "CUDAException.h"
#include "DispatchTableEntry.h"

#if defined(PGA_CORE_EXPORT)
#    if (PGA_CORE_EXPORT == 1)
#        define PGA_CORE_API(__returnType__) __declspec(dllexport) __returnType__ __stdcall
#    else
#        define PGA_CORE_API(__returnType__) __returnType__ __cdecl
#    endif
namespace PGA
{
	namespace Device
	{
		__device__ DispatchTableEntry* DispatchTable;
		__device__ unsigned int NumEntries;
		__device__ math::float3 CameraPosition;
		__device__ __constant__ float Seeds[PGA::Constants::MaxNumAxioms];

		PGA_CORE_API(void) setSeeds(float* seeds, unsigned int numSeeds)
		{
			auto size = (numSeeds % (PGA::Constants::MaxNumAxioms + 1)) * sizeof(float);
			PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::Seeds, seeds, size));
		}

	}

	namespace Host
	{
		DispatchTableEntry* DispatchTable;
		unsigned int NumEntries;
		math::float3 CameraPosition;
		float Seeds[PGA::Constants::MaxNumAxioms];

		PGA_CORE_API(void) setSeeds(float* seeds, unsigned int numSeeds)
		{
			auto size = (numSeeds % (PGA::Constants::MaxNumAxioms + 1)) * sizeof(float);
			memcpy(Host::Seeds, seeds, size);
		}

	}

	namespace GlobalVars
	{
		__host__ __device__ __inline__ DispatchTableEntry& getDispatchTableEntry(int i)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (i < 0 || i >= Device::NumEntries)
			{
				printf("PGA::GlobalVars::getDispatchTableEntry(..): invalid dispatch table entry index [Device::NumEntries=%d, i=%d] (CUDA thread %d %d)", Device::NumEntries, i, threadIdx.x, blockIdx.x);
				asm("trap;");
			}
#endif
			return Device::DispatchTable[i];
#else
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (i < 0 || i >= Host::NumEntries)
				throw std::runtime_error("PGA::GlobalVars::getDispatchTableEntry(..): invalid dispatch table entry index [Host::NumEntries=" + std::to_string(Host::NumEntries) + ", i=" + std::to_string(i) + "]");
#endif
			return Host::DispatchTable[i];
#endif
		}

		__host__ __device__ __inline__ unsigned int getNumEntries()
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return Device::NumEntries;
#else
			return Host::NumEntries;
#endif
		}

		__host__ __device__ __inline__ math::float3 getCameraPosition()
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return Device::CameraPosition;
#else
			return Host::CameraPosition;
#endif
		}

		__host__ __device__ float getSeed(unsigned int axiomIndex)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return Device::Seeds[axiomIndex % PGA::Constants::MaxNumAxioms];
#else
			return Host::Seeds[axiomIndex % PGA::Constants::MaxNumAxioms];
#endif
		}

	}
}
#else
#    if (defined(PGA_CORE_IMPORT) && (PGA_CORE_IMPORT == 1))
#		define PGA_CORE_API(__returnType__) __declspec(dllimport) __returnType__ __stdcall
#	 else
#		define PGA_CORE_API(__returnType__) extern __returnType__ __cdecl
#	 endif
namespace PGA
{
	namespace Device
	{
		PGA_CORE_API(void) setSeeds(float* seeds, unsigned int numSeeds);

	}

	namespace Host
	{
		PGA_CORE_API(void) setSeeds(float* seeds, unsigned int numSeeds);

	}

}

#endif

