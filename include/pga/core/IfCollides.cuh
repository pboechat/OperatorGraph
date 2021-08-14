#pragma once

#include "BVHTraversal.cuh"
#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "SymbolDecorator.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/vector.h>

#include <cstdio>
#include <string>

namespace PGA
{
	namespace Operators
	{
		template <typename ColliderTagT, typename NextOperatorIfTrueT, typename NextOperatorIfFalseT>
		class IfCollides
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				unsigned int tag = static_cast<unsigned int>(ColliderTagT::eval(symbol));

				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::IfCollides>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing IfCollides with params: ColliderTagId=%d, NextOperatorIfTrueT=%s, NextOperatorIfFalseT=%s\n", tag, NextOperatorIfTrueT::name(), NextOperatorIfFalseT::name());
#else
					std::cout << "Executing IfCollides with params: ColliderTagId=" << tag << ", NextOperatorIfTrueT=" << NextOperatorIfTrueT::name() << ", NextOperatorIfFalseT=" << NextOperatorIfFalseT::name() << std::endl;
#endif

				if (ContextSensitivity::BVHTraversal::checkCollision(tag, *symbol))
				{
					if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::IfCollides>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Collision detected\n");
#else
						std::cout << "Collision detected" << std::endl;
#endif
					SymbolDecorator<NextOperatorIfTrueT>::run(symbol, symbol);
					NextOperatorIfTrueT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}
				else
				{
					if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::IfCollides>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("No collision detected\n");
#else
						std::cout << "No collision detected" << std::endl;
#endif
					SymbolDecorator<NextOperatorIfFalseT>::run(symbol, symbol);
					NextOperatorIfFalseT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "IfCollides";
			}

			__host__ __inline__ static std::string toString()
			{
				return "IfCollides<" + ColliderTagT::toString() + ", " + NextOperatorIfTrueT::toString() + ", " + NextOperatorIfFalseT::toString() + ">";
			}

		};

	}

}
