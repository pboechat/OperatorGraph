#pragma once

#include <cstdio>
#include <string>
#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Axis.h"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "Parameters.cuh"
#include "TStdLib.h"
#include "SymbolDecorator.cuh"

namespace PGA
{
	namespace Operators
	{
		template <typename XT, typename YT, typename ZT, typename NextOperatorT>
		class SwapSize
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto x = static_cast<PGA::Axis>(__float2uint_rd(XT::eval(symbol)));
				auto y = static_cast<PGA::Axis>(__float2uint_rd(YT::eval(symbol)));
				auto z = static_cast<PGA::Axis>(__float2uint_rd(ZT::eval(symbol)));
#else
				auto x = static_cast<PGA::Axis>(static_cast<unsigned int>(XT::eval(symbol)));
				auto y = static_cast<PGA::Axis>(static_cast<unsigned int>(YT::eval(symbol)));
				auto z = static_cast<PGA::Axis>(static_cast<unsigned int>(ZT::eval(symbol)));
#endif
				const auto& size = symbol->getSize();
				float sx = size[x], sy = size[y], sz = size[z];
				symbol->setSize(math::float3(sx, sy, sz));
				SymbolDecorator<NextOperatorT>::run(symbol, symbol);
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "SwapSize";
			}

			__host__ __inline__ static std::string toString()
			{
				return "SwapSize<" + XT::toString() + ", " + YT::toString() + ", " + ZT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

	}

}
