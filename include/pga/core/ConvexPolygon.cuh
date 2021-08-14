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
		template <unsigned int NumSidesT, bool Regular>
		class ConvexPolygon : public Shape
		{
		private:
			__host__ __device__ __inline__ void copyVertices(const math::float2 otherVertices[NumSidesT])
			{
				// TODO: optimize
				for (auto i = 0; i < NumSidesT; i++)
					vertices[i] = otherVertices[i];
			}

		public:
			static_assert(NumSidesT >= 3, "Convex Polygon cannot have less than 3 sides");
			static const unsigned int NumSides = NumSidesT;

			math::float2 vertices[NumSidesT];

			__host__ __device__ ConvexPolygon() {}

			__host__ __device__ ConvexPolygon(const math::float2* vertices, unsigned int numSides)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (numSides != NumSides)
				{
					printf("irregular convex polygon num. sides must be exactly %d (numSides=%d)\n", NumSides, numSides);
					asm("trap;");
				}
#endif
#else
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL == 1 || PGA_INVARIANT_CHECKING_LVL == 3)
				if (numSides != NumSides)
					throw std::runtime_error(("irregular convex polygon num. sides must be exactly " + std::to_string(NumSides) + " (numSides=" + std::to_string(numSides) + ")").c_str());
#endif
#endif
				for (unsigned int i = 0; i < NumSides; i++)
					this->vertices[i] = vertices[i];
			}

			__host__ __device__ ConvexPolygon(const ConvexPolygon& other) : Shape(other) 
			{
				copyVertices(other.vertices);
			}

			__host__ __device__ __inline__ math::float2 getVertex(unsigned int i)
			{
				return vertices[i];
			}

			__host__ __device__ __inline__ ConvexPolygon& operator=(const ConvexPolygon& other)
			{
				Shape::operator=(other);
				copyVertices(other.vertices);
				return *this;
			}

			__host__ __inline__ static std::string toString()
			{
				return "ConvexPolygon<" + std::to_string(NumSidesT) + ", false>";
			}

		};

		template <unsigned int NumSidesT>
		struct ConvexPolygon<NumSidesT, true> : public Shape
		{
			static_assert(NumSidesT >= 3, "Convex Polygon cannot have less than 3 sides");
			static const unsigned int NumSides = NumSidesT;
			bool invert;

			// NOTE: doesn't need to have explicit vertices

			__host__ __device__ ConvexPolygon() : invert(false) {}
			__host__ __device__ ConvexPolygon(const ConvexPolygon& other) : Shape(other), invert(other.invert) {}

			__host__ __device__ __inline__ math::float2 getVertex(unsigned int i)
			{
				float sign = 1.0f - 2.0f * invert;
				float angle = 6.2831853071795864769252868f / (float)NumSidesT * i + 1.5707963267948966192313217f;
				return math::float2(cos(angle) * sign, sin(angle) * sign) * 0.5f;
			}

			__host__ __device__ ConvexPolygon& operator=(const ConvexPolygon& other)
			{
				Shape::operator=(other);
				invert = other.invert;
				return *this;
			}

			__host__ __inline__ static std::string toString()
			{
				return "ConvexPolygon<" + std::to_string(NumSidesT) + ", true>";
			}
		};

		typedef ConvexPolygon<3, true> Triangle;
		typedef ConvexPolygon<4, true> Quad;
		typedef ConvexPolygon<5, true> Pentagon;
		typedef ConvexPolygon<6, true> Hexagon;
		typedef ConvexPolygon<7, true> Heptagon;
		typedef ConvexPolygon<8, true> Octagon;

		template <unsigned int NumSides>
		struct GetName < ConvexPolygon<NumSides, false> >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "NonRegularConvexPolygon";
			}

		};

		template <unsigned int NumSides>
		struct GetName < ConvexPolygon<NumSides, true> >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "RegularConvexPolygon";
			}

		};

		template <>
		struct GetName < Triangle >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Triangle";
			}

		};

		template <>
		struct GetName < Quad >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Quad";
			}

		};

		template <>
		struct GetName < Pentagon >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Pentagon";
			}

		};

		template <>
		struct GetName < Hexagon >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Hexagon";
			}

		};

		template <>
		struct GetName < Heptagon >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Heptagon";
			}

		};

		template <>
		struct GetName < Octagon >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Octagon";
			}

		};

		template <unsigned int NumSides, bool Regular>
		struct IsPlanar < ConvexPolygon<NumSides, Regular> >
		{
			static const bool Result = true;

		};

	}

}
