#pragma once

#include <cstdio>
#include <string>
#include <cuda_runtime_api.h>

#include <math/vector.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "TStdLib.h"
#include "SymbolDecorator.cuh"
#include "IntermediateSymbolsBufferAdapter.cuh"

namespace PGA
{
	namespace Operators
	{
		template <typename ColliderTagT, typename NextOperatorT>
		class Collider
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto tag = __float2uint_rd(ColliderTagT::eval(symbol));
#else
				auto tag = static_cast<unsigned int>(ColliderTagT::eval(symbol));
#endif

				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Collider>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing collider with param: ColliderTagT=%d, NextOperatorT=%s\n", tag, NextOperatorT::name());
#else
					std::cout << "Executing collider with param: ColliderTagT=" << tag << ", NextOperatorT=" << NextOperatorT::name() << std::endl;
#endif
				if (tag > 0)
				{
					ContextSensitivity::PerShapeIntermediateSymbolsBufferAdapter<ShapeT>::store(tag, *symbol);
					if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Collider>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("IntermediateSymbolBuffer counter=%d\n", ContextSensitivity::PerShapeIntermediateSymbolsBufferAdapter<ShapeT>::getCounter());
#else
						std::cout << "IntermediateSymbolBuffer counter=" << ContextSensitivity::PerShapeIntermediateSymbolsBufferAdapter<ShapeT>::getCounter() << std::endl;
#endif
				}
				SymbolDecorator<NextOperatorT>::run(symbol, symbol);
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Collider";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Collider<" + ColliderTagT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

	}

}
