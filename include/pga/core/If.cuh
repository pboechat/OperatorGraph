#pragma once

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
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
		template <typename ExpressionT, typename NextOperatorIfTrueT, typename NextOperatorIfFalseT>
		class If
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				float result = ExpressionT::eval(symbol);
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::IfSizeLess>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing if with params: result=%f, nextOperatorIfTrue=%d, nextOperatorIfFalse=%d\n", result, NextOperatorIfTrueT::name(), NextOperatorIfFalseT::name());
#else
					std::cout << "Executing if with params: result=" << result << ", nextOperatorIfTrue=" << NextOperatorIfTrueT::name() << ", nextOperatorIfFalse=" << NextOperatorIfFalseT::name() << std::endl;
#endif
				if (result)
				{
					SymbolDecorator<NextOperatorIfTrueT>::run(symbol, symbol);
					NextOperatorIfTrueT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}
				else
				{
					SymbolDecorator<NextOperatorIfFalseT>::run(symbol, symbol);
					NextOperatorIfFalseT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "If";
			}

			__host__ __inline__ static std::string toString()
			{
				return "If<" + ExpressionT::toString() + ", " + NextOperatorIfTrueT::toString() + ", " + NextOperatorIfFalseT::toString() + ">";
			}

		};

		template <typename AxisT, typename SizeT, typename NextOperatorIfTrueT, typename NextOperatorIfFalseT>
		class IfSizeLess
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto axis = static_cast<PGA::Axis>(__float2uint_rd(AxisT::eval(symbol)));
#else
				auto axis = static_cast<PGA::Axis>(static_cast<unsigned int>(AxisT::eval(symbol)));
#endif
				auto size = SizeT::eval(symbol);
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::IfSizeLess>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing if size less with params: axis=%d, size=%f, nextOperatorIfTrue=%d, nextOperatorIfFalse=%d\n", axis, size, NextOperatorIfTrueT::name(), NextOperatorIfFalseT::name());
#else
					std::cout << "Executing if size less with params: axis=" << axis << ", size=" << size << ", nextOperatorIfTrue=" << NextOperatorIfTrueT::name() << ", nextOperatorIfFalse=" << NextOperatorIfFalseT::name() << std::endl;
#endif
				if (symbol->getSize()[axis] < size)
				{
					SymbolDecorator<NextOperatorIfTrueT>::run(symbol, symbol);
					NextOperatorIfTrueT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}
				else
				{
					SymbolDecorator<NextOperatorIfFalseT>::run(symbol, symbol);
					NextOperatorIfFalseT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "IfSizeLess";
			}

			__host__ __inline__ static std::string toString()
			{
				return "IfSizeLess<" + AxisT::toString() + ", " + SizeT::toString() + ", " + NextOperatorIfTrueT::toString() + ", " + NextOperatorIfFalseT::toString() + ">";
			}

		};

	}

}
