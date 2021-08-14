#pragma once

#include "Shape.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/vector.h>

#include <stdexcept>
#include <string>

namespace PGA
{
	namespace Shapes
	{
		template <unsigned int MaxNumVerticesT, bool ConvexT>
		class DynamicPolygon : public Shape
		{
		private:
			__host__ __device__ __inline__ void copyVertices(const math::float2 otherVertices[MaxNumVerticesT])
			{
				// TODO: optimize
				for (auto i = 0; i < MaxNumVerticesT; i++)
					vertices[i] = otherVertices[i];
			}

		public:
			math::float2 vertices[MaxNumVerticesT];
			unsigned int numSides;
			bool invert;

			__host__ __device__ DynamicPolygon() : numSides(0), invert(false) {}

			__host__ __device__ DynamicPolygon(const math::float2* vertices, unsigned int numSides)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (numSides < 3)
				{
					printf("dynamic polygon num. sides must be greater than or equal 3 (numSides=%d)\n", numSides);
					asm("trap;");
				}
#endif
#else
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL == 1 || PGA_INVARIANT_CHECKING_LVL == 3)
				if (numSides < 3)
					throw std::runtime_error(("dynamic polygon num. sides must be greater than or equal 3 (numSides=" + std::to_string(numSides) + ")").c_str());
#endif
#endif
				this->numSides = numSides;
				invert = false;
				for (unsigned int i = 0; i < numSides; i++)
					this->vertices[i] = vertices[i];
			}

			__host__ __device__ DynamicPolygon(const DynamicPolygon<MaxNumVerticesT, ConvexT>& other) : Shape(other), numSides(other.numSides), invert(other.invert)
			{
				copyVertices(other.vertices);
			}

			__host__ __device__ __inline__ unsigned int getNumSides() const
			{
				return numSides;
			}

			__host__ __device__ __inline__ math::float4x4 getAdjustedModel() const
			{
				float sign = (!invert) + (invert * -1.0f);
				return math::float4x4(sign * model._11, sign * model._12, model._13, model._14,
									  sign * model._21, sign * model._22, model._23, model._24,
									  sign * model._31, sign * model._32, model._33, model._34,
									  0.0f, 0.0f, 0.0f, 1.0f);
			}

			__host__ __device__ __inline__ math::float2 getVertex(unsigned int i) const
			{
				if (invert)
				{
					math::float2 vertex = vertices[numSides - i - 1];
					vertex.y = -vertex.y;
					return vertex;
				}
				else
				{
					return vertices[i];
				}
			}

			__host__ __device__ __inline__ DynamicPolygon<MaxNumVerticesT, ConvexT>& operator=(const DynamicPolygon<MaxNumVerticesT, ConvexT>& other)
			{
				Shape::operator=(other);
				numSides = other.numSides;
				invert = other.invert;
				copyVertices(other.vertices);
				return *this;
			}

			__host__ __inline__ static std::string toString()
			{
				return "DynamicPolygon<" + std::to_string(MaxNumVerticesT) + ", " + std::to_string(ConvexT) + ">";
			}

		};

		template <unsigned int MaxNumVerticesT>
		struct GetName < DynamicPolygon<MaxNumVerticesT, true> >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "DynamicConvexPolygon";
			}

		};

		template <unsigned int MaxNumVerticesT>
		struct GetName < DynamicPolygon<MaxNumVerticesT, false> >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "DynamicPolygon";
			}

		};

		template <unsigned int MaxNumVerticesT, bool ConvexT>
		struct IsPlanar < DynamicPolygon<MaxNumVerticesT, ConvexT> >
		{
			static const bool Result = true;

		};

	}

}
