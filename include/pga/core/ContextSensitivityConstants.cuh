#pragma once

#include <math/vector.h>

#include "Shapes.cuh"

namespace PGA
{
	namespace ContextSensitivity
	{
		namespace Constants
		{
			const unsigned int MaxBVHDepth = 16;

			const math::float3 SceneBoundaryMin(-100.0f, -100.0f, -100.0f);
			const math::float3 SceneBoundaryMax(1000.0f, 1000.0f, 1000.0f);

			template <typename Shape>
			struct GetMaxIntermediateSymbols;

			template<>
			struct GetMaxIntermediateSymbols < Shapes::Box >
			{
				static const unsigned int Result = 100;

			};

			template<>
			struct GetMaxIntermediateSymbols < Shapes::Quad >
			{
				static const unsigned int Result = 1;

			};

			template<>
			struct GetMaxIntermediateSymbols < Shapes::Sphere >
			{
				static const unsigned int Result = 1;

			};

			template<unsigned int NumSides, bool Regular>
			struct GetMaxIntermediateSymbols < Shapes::ConvexPolygon<NumSides, Regular> >
			{
				static const unsigned int Result = 1;

			};

			template<unsigned int NumSides, bool Regular>
			struct GetMaxIntermediateSymbols < Shapes::ConvexRightPrism<NumSides, Regular> >
			{
				static const unsigned int Result = 1;

			};

		}

	}

}
