#pragma once

#include "Axis.h"
#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
#include "RepeatMode.h"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "SymbolDecorator.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/vector.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace PGA
{
	namespace Operators
	{
		namespace
		{
			template <typename ShapeT>
			__host__ __device__ __inline__ float getNewExtent(PGA::Axis axis, float extent, const ShapeT& shape)
			{
				return ((extent < 0) * shape.getSize()[axis] * abs(extent)) + ((extent > 0) * extent);
			}

		}

		template <bool ParallelT /* true */, typename AxisT, typename ExtentT, typename RepeatModeT, typename NextOperatorT>
		class Repeat
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				// TODO: implement
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Operators::Repeat::execute(): operator not implemented [CUDA thread %d %d]\n", threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error("PGA::Operators::Repeat::execute(): operator not implemented");
#endif
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Repeat";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Repeat<true, " + AxisT::toString() + ", " + ExtentT::toString() + ", " + RepeatModeT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

		template <typename AxisT, typename ExtentT, typename RepeatModeT, typename NextOperatorT>
		class Repeat < false, AxisT, ExtentT, RepeatModeT, NextOperatorT >
		{
		public:
			static const int NumThreadsT = 1;

			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				auto axis = static_cast<PGA::Axis>(static_cast<unsigned int>(AxisT::eval(symbol)));
				auto repeatMode = static_cast<PGA::RepeatMode>(static_cast<unsigned int>(RepeatModeT::eval(symbol)));
				auto extent = getNewExtent(axis, ExtentT::eval(symbol), *symbol);
				if (extent <= 0.0f)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::Repeat::execute(): extent parameter cannot be less than or 0 [CUDA thread %d %d]\n", threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error("PGA::Operators::Repeat::execute(): extent parameter cannot be less than or 0");
#endif
				}
				auto size = symbol->getSize();
				auto totalExtent = size[axis];
				if (totalExtent < 0.0f)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::Repeat::execute(): shape size being subdivided is negative! [CUDA thread %d %d]\n", threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error("PGA::Operators::Repeat::execute(): shape size being subdivided is negative!");
#endif
				}
				auto division = totalExtent / extent;
				float rest;
				bool exact;
				unsigned int repetitions;
				float extent1, extent2, extent3;
				switch (repeatMode)
				{
				case PGA::ANCHOR_TO_START:
				case PGA::ANCHOR_TO_END:
					rest = fmodf(totalExtent, extent);
					exact = (rest == 0);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					repetitions = __float2uint_ru(division);
#else
					repetitions = static_cast<unsigned int>(ceilf(division));
#endif
					extent1 = (exact) * extent + (!exact) * rest;
					extent2 = extent3 = extent;
					break;
				case PGA::ADJUST_TO_FILL:
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					repetitions = __float2uint_rd(division);
					if (repetitions > 0)
						extent1 = extent2 = extent3 = totalExtent / __uint2float_rd(repetitions);
					else
						extent1 = extent2 = extent3 = 0;
#else
					repetitions = static_cast<unsigned int>(floorf(division));
					if (repetitions > 0)
						extent1 = extent2 = extent3 = totalExtent / static_cast<float>(repetitions);
					else
						extent1 = extent2 = extent3 = 0;
#endif
					break;
				default:
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::Repeat::execute(): unknown PGA::RepeatMode [CUDA thread %d %d]\n", threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error("PGA::Operators::Repeat::execute(): unknown PGA::RepeatMode");
#endif
				}
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Repeat>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing repeat %d times (axis=%d, repeatMode=%d, totalExtent=%.3f, extent=%.3f) [CUDA thread %d %d]\n", repetitions, axis, repeatMode, totalExtent, extent, threadIdx.x, blockIdx.x);
#else
					std::cout << "Executing repeat " << repetitions << " times (axis=" << axis << ", repeatMode=" << repeatMode << ", totalExtent=" << totalExtent << ", extent=" << extent << "): " << std::endl;
#endif
				}
				bool start = repeatMode == PGA::ANCHOR_TO_START;
				auto startExtent = (start) * extent1 + (!start) * extent3;
				Symbol<ShapeT> newSymbol(*symbol);
				math::float3 newSize(size);
				newSize[axis] = startExtent;
				newSymbol.setSize(newSize);
				auto model = symbol->getModel4();
				math::float3 offset(0.0f);
				auto halfStartExtent = startExtent * 0.5f;
				offset[axis] = (totalExtent * -0.5f) + halfStartExtent;
				newSymbol.setModel(model * math::float4x4::translate(offset));
				newSymbol.setSeed(symbol->generateNextSeed(1.0f));
				SymbolDecorator<NextOperatorT>::run(symbol, &newSymbol);
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
				offset[axis] += halfStartExtent;
				auto halfMidExtent = extent2 * 0.5f;
				newSize[axis] = extent2;
				for (auto i = 2u; i < repetitions; i++)
				{
					offset[axis] += halfMidExtent;
					newSymbol.setSize(newSize);
					newSymbol.setModel(model * math::float4x4::translate(offset));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					newSymbol.setSeed(symbol->generateNextSeed(__uint2float_rd(i)));
#else
					newSymbol.setSeed(symbol->generateNextSeed(static_cast<float>(i)));
#endif
					SymbolDecorator<NextOperatorT>::run(symbol, &newSymbol);
					NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
					offset[axis] += halfMidExtent;
				}
				if (repetitions > 1)
				{
					auto endExtent = (start) * extent3 + (!start) * extent1;
					offset[axis] += endExtent * 0.5f;
					newSize[axis] = endExtent;
					newSymbol.setSize(newSize);
					newSymbol.setModel(model * math::float4x4::translate(offset));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					newSymbol.setSeed(symbol->generateNextSeed(__uint2float_rd(repetitions)));
#else
					newSymbol.setSeed(symbol->generateNextSeed(static_cast<float>(repetitions)));
#endif
					SymbolDecorator<NextOperatorT>::run(symbol, &newSymbol);
					NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Repeat";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Repeat<false, " + AxisT::toString() + ", " + ExtentT::toString() + ", " + RepeatModeT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

	}

}
