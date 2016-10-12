#pragma once

#include <cuda_runtime_api.h>

#include <math/vector.h>

#include "ContextSensitivityConstants.cuh"
#include "IntermediateSymbol.cuh"

namespace PGA
{
	namespace ContextSensitivity
	{
		template <typename ShapeT>
		struct IntermediateSymbolsBuffer
		{
			unsigned int counter;
			IntermediateSymbol<ShapeT>* shapes;

			__host__ __device__ __inline__ void store(const IntermediateSymbol<ShapeT>& intermediateSymbol)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				unsigned int i = atomicAdd(&counter, 1);
#else
				unsigned int i = counter++;
#endif
				shapes[i % ContextSensitivity::Constants::GetMaxIntermediateSymbols<ShapeT>::Result] = intermediateSymbol;
			}

		};

	}

}
