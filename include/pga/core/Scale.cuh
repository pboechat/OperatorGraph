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
#include "Parameters.cuh"
#include "TStdLib.h"
#include "SymbolDecorator.cuh"

namespace PGA
{
	namespace Operators
	{
		template <typename XT, typename YT, typename ZT, typename NextOperatorT>
		class Scale
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				float x = XT::eval(symbol);
				float y = YT::eval(symbol);
				float z = ZT::eval(symbol);
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Scale>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing scale with params: x=%f, y=%f, z=%f\n", x, y, z);
#else
					std::cout << "Executing scale with params: x=" << x << ", y=" << y << ", z=" << z << std::endl;
#endif
				math::float3 size = symbol->getSize();
				symbol->setSize(math::float3(((x < 0) * size.x * x * -1) + ((x > 0) * x),
					((y < 0) * size.y * y * -1) + ((y > 0) * y),
					((z < 0) * size.z * z * -1) + ((z > 0) * z)));
				SymbolDecorator<NextOperatorT>::run(symbol, symbol);
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Scale";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Scale<" + XT::toString() + ", " + YT::toString() + ", " + ZT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

	}

}
