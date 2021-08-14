#pragma once

#include "InstancedTriangleMeshData.h"
#include "RenderingConstants.h"
#include "TriangleMeshData.h"

#include <cuda_runtime_api.h>
#include <pga/core/CUDAException.h>

#include <memory>
#include <stdexcept>

#if defined(PGA_RENDERING_EXPORT)
#    if (PGA_RENDERING_EXPORT == 1)
#        define PGA_RENDERING_API(__returnType__) __declspec(dllexport) __returnType__ __stdcall
#    else
#        define PGA_RENDERING_API(__returnType__) __returnType__ __cdecl
#    endif

namespace PGA
{
	namespace Rendering
	{
		namespace Device
		{
			__device__ __constant__ unsigned int NumTriangleMeshes;
			__device__ __constant__ unsigned int NumInstancedTriangleMeshes;
			__device__ TriangleMeshData TriangleMeshes[Constants::MaxNumTriangleMeshes];
			__device__ InstancedTriangleMeshData InstancedTriangleMeshes[Constants::MaxNumInstancedTriangleMeshes];

			PGA_RENDERING_API(void) setNumTriangleMeshes(unsigned int numTriangleMeshes)
			{
				CUDA_CHECKED_CALL(cudaMemcpyToSymbol(NumTriangleMeshes, (void*)&numTriangleMeshes, sizeof(unsigned int)));
			}

			PGA_RENDERING_API(void) setNumInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes)
			{
				CUDA_CHECKED_CALL(cudaMemcpyToSymbol(NumInstancedTriangleMeshes, (void*)&numInstancedTriangleMeshes, sizeof(unsigned int)));
			}

			PGA_RENDERING_API(void) getTriangleMeshes(unsigned int numTriangleMeshes, std::unique_ptr<TriangleMeshData[]>& triangleMeshesData)
			{
				CUDA_CHECKED_CALL(cudaMemcpyFromSymbol(triangleMeshesData.get(), TriangleMeshes, numTriangleMeshes * sizeof(TriangleMeshData)));
			}

			PGA_RENDERING_API(void) setTriangleMeshes(unsigned int numTriangleMeshes, const std::unique_ptr<TriangleMeshData[]>& triangleMeshesData)
			{
				CUDA_CHECKED_CALL(cudaMemcpyToSymbol(TriangleMeshes, triangleMeshesData.get(), numTriangleMeshes * sizeof(TriangleMeshData)));
			}

			PGA_RENDERING_API(void) getInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData)
			{
				CUDA_CHECKED_CALL(cudaMemcpyFromSymbol(instancedTriangleMeshesData.get(), InstancedTriangleMeshes, numInstancedTriangleMeshes * sizeof(InstancedTriangleMeshData)));
			}

			PGA_RENDERING_API(void) setInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, const std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData)
			{
				CUDA_CHECKED_CALL(cudaMemcpyToSymbol(InstancedTriangleMeshes, instancedTriangleMeshesData.get(), numInstancedTriangleMeshes * sizeof(InstancedTriangleMeshData)));
			}

		}

		namespace Host
		{
			unsigned int NumTriangleMeshes = 0;
			unsigned int NumInstancedTriangleMeshes = 0;
			TriangleMeshData TriangleMeshes[Constants::MaxNumTriangleMeshes];
			InstancedTriangleMeshData InstancedTriangleMeshes[Constants::MaxNumInstancedTriangleMeshes];

			PGA_RENDERING_API(void) setNumTriangleMeshes(unsigned int numTriangleMeshes)
			{
				Host::NumTriangleMeshes = numTriangleMeshes;
			}

			PGA_RENDERING_API(void) setNumInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes)
			{
				Host::NumInstancedTriangleMeshes = numInstancedTriangleMeshes;
			}

			PGA_RENDERING_API(void) getTriangleMeshes(unsigned int numTriangleMeshes, std::unique_ptr<TriangleMeshData[]>& triangleMeshesData)
			{
				memcpy(triangleMeshesData.get(), Host::TriangleMeshes, numTriangleMeshes * sizeof(TriangleMeshData));
			}

			PGA_RENDERING_API(void) setTriangleMeshes(unsigned int numTriangleMeshes, const std::unique_ptr<TriangleMeshData[]>& triangleMeshesData)
			{
				memcpy(Host::TriangleMeshes, triangleMeshesData.get(), numTriangleMeshes * sizeof(TriangleMeshData));
			}

			PGA_RENDERING_API(void) getInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData)
			{
				memcpy(instancedTriangleMeshesData.get(), Host::InstancedTriangleMeshes, numInstancedTriangleMeshes * sizeof(InstancedTriangleMeshData));
			}

			PGA_RENDERING_API(void) setInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, const std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData)
			{
				memcpy(Host::InstancedTriangleMeshes, instancedTriangleMeshesData.get(), numInstancedTriangleMeshes * sizeof(InstancedTriangleMeshData));
			}
		}

		namespace GlobalVars
		{
			__host__ __device__ __inline__ unsigned int getNumTriangleMeshes()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return Device::NumTriangleMeshes;
#else
				return Host::NumTriangleMeshes;
#endif
			}

			__host__ __device__ __inline__ unsigned int getNumInstancedTriangleMeshes()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return Device::NumInstancedTriangleMeshes;
#else
				return Host::NumInstancedTriangleMeshes;
#endif
			}

			__host__ __device__ __inline__ TriangleMeshData& getTriangleMesh(unsigned int i)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (i >= Device::NumTriangleMeshes)
				{
					printf("PGA::GlobalVars::getTriangleMesh(..): invalid triangle mesh index [Device::NumTriangleMeshes=%d, i=%d] (CUDA thread %d %d)", Device::NumTriangleMeshes, i, threadIdx.x, blockIdx.x);
					asm("trap;");
				}
#endif
				return Device::TriangleMeshes[i];
#else
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (i >= Host::NumTriangleMeshes)
					throw std::runtime_error("PGA::GlobalVars::getTriangleMesh(..): invalid triangle mesh index [Host::NumTriangleMeshes=" + std::to_string(Host::NumTriangleMeshes) + ", i=" + std::to_string(i) + "]");
#endif
				return Host::TriangleMeshes[i];
#endif
			}

			__host__ __device__ __inline__ InstancedTriangleMeshData& getInstancedTriangleMesh(unsigned int i)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (i >= Device::NumInstancedTriangleMeshes)
				{
					printf("PGA::GlobalVars::getInstancedTriangleMesh(..): invalid instanced triangle mesh index [Device::NumInstancedTriangleMeshes=%d, i=%d] (CUDA thread %d %d)", Device::NumTriangleMeshes, i, threadIdx.x, blockIdx.x);
					asm("trap;");
				}
#endif
				return Device::InstancedTriangleMeshes[i];
#else
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (i >= Host::NumInstancedTriangleMeshes)
					throw std::runtime_error("PGA::GlobalVars::getInstancedTriangleMesh(..): invalid instanced triangle mesh index [Host::NumInstancedTriangleMeshes=" + std::to_string(Host::NumInstancedTriangleMeshes) + ", i=" + std::to_string(i) + "]");
#endif
				return Host::InstancedTriangleMeshes[i];
#endif
			}

		}

	}

}

#else
#    if (defined(PGA_RENDERING_IMPORT) && (PGA_RENDERING_IMPORT == 1))
#		define PGA_RENDERING_API(__returnType__) __declspec(dllimport) __returnType__ __stdcall
#	 else
#		define PGA_RENDERING_API(__returnType__) extern __returnType__ __cdecl
#	 endif
namespace PGA
{
	namespace Rendering
	{
		namespace Device
		{
			PGA_RENDERING_API(void) setNumTriangleMeshes(unsigned int numTriangleMeshes);
			PGA_RENDERING_API(void) setNumInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes);
			PGA_RENDERING_API(void) getTriangleMeshes(unsigned int numTriangleMeshes, std::unique_ptr<TriangleMeshData[]>& triangleMeshesData);
			PGA_RENDERING_API(void) setTriangleMeshes(unsigned int numTriangleMeshes, const std::unique_ptr<TriangleMeshData[]>& triangleMeshesData);
			PGA_RENDERING_API(void) getInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData);
			PGA_RENDERING_API(void) setInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, const std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData);

		}

		namespace Host
		{
			PGA_RENDERING_API(void) setNumTriangleMeshes(unsigned int numTriangleMeshes);
			PGA_RENDERING_API(void) setNumInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes);
			PGA_RENDERING_API(void) getTriangleMeshes(unsigned int numTriangleMeshes, std::unique_ptr<TriangleMeshData[]>& triangleMeshesData);
			PGA_RENDERING_API(void) setTriangleMeshes(unsigned int numTriangleMeshes, const std::unique_ptr<TriangleMeshData[]>& triangleMeshesData);
			PGA_RENDERING_API(void) getInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData);
			PGA_RENDERING_API(void) setInstancedTriangleMeshes(unsigned int numInstancedTriangleMeshes, const std::unique_ptr<InstancedTriangleMeshData[]>& instancedTriangleMeshesData);

		}

	}

}
#endif