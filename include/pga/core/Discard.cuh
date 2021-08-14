#pragma once

#include "DebugFlags.h"
#include "Shapes.cuh"
#include "Symbol.cuh"

#include <cuda_runtime_api.h>

#include <string>

namespace PGA
{
	namespace Operators
	{
		class Discard
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Discard";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Discard";
			}

		};

	}

}
