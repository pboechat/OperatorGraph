#pragma once

#include <cstdio>
#include <string>
#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
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
		class Translate
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				float x = XT::eval(symbol);
				float y = YT::eval(symbol);
				float z = ZT::eval(symbol);
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Translate>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing translate with params: x=%f, y=%f, z=%f\n", x, y, z);
#else
					std::cout << "Executing translate with params: x=" << x << ", y=" << y << ", z=" << z << std::endl;
#endif
				symbol->setModel(symbol->getModel4() * math::float4x4::translate(math::float3(x, y, z)));
				SymbolDecorator<NextOperatorT>::run(symbol, symbol);
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Translate";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Translate<" + XT::toString() + ", " + YT::toString() + ", " + ZT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

	}

}
