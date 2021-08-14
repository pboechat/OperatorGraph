#pragma once

#include "Axis.h"
#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
#include "Shapes.cuh"
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
		// NOTE: not using axis atm
		template <typename AxisT, typename ExtentT, typename NextOperatorT>
		class Extrude
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				float extent = ExtentT::eval(symbol);
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Extrude>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing extrude with extent=%f\n", extent);
#else
					std::cout << "Executing extrude with extent=" << extent << std::endl;
#endif
				typedef typename Shapes::GetExtrudedType<ShapeT>::Result ExtrudedShape;
				// NOTE: It's necessary to create a new symbol, because it has a different shape type
				Symbol<ExtrudedShape> newSymbol(ExtrudedShape::fromPlanarType(*symbol));
				SymbolDecorator<NextOperatorT>::run(symbol, &newSymbol);
				math::float4x4 model = symbol->getModel4();
				math::float3 size = symbol->getSize();
				// NOTE: Planar shapes point towards positive Z, whilst extruded shapes point towards positive Y
				newSymbol.setModel(model *
					math::float4x4(
						1.0f, 0.0f, 0.0f, 0.0f,
						0.0f, 0.0f, -1.0f, 0.0f,
						0.0f, 1.0f, 0.0f, 0.0f,
						0.0f, 0.0f, 0.0f, 1.0f
					)
				);
				newSymbol.setSize(math::float3(size.x, extent, size.y));
				newSymbol.setSeed(symbol->generateNextSeed(0));
				newSymbol.setCustomAttribute(symbol->getCustomAttribute());
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Extrude";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Extrude<" + AxisT::toString() + ", " + ExtentT::toString() + ", " + NextOperatorT::toString() + ">";
			}

		};

	}

}
