#pragma once

#include <string>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <math/vector.h>

#include "Shape.cuh"
#include "ConvexPolygon.cuh"
#include "TStdLib.h"

namespace PGA
{
	namespace Shapes
	{
		template <unsigned int NumSidesT, bool RegularT>
		class ConvexRightPrism : public Shape
		{
		private:
			__host__ __device__ __inline__ void copyVertices(const math::float2 otherVertices[NumSidesT])
			{
				// TODO: optimize
				for (auto i = 0; i < NumSidesT; i++)
					vertices[i] = otherVertices[i];
			}

		public:
			static_assert(NumSidesT >= 3, "convex right prism cannot have less than 3 side faces");
			static const unsigned int NumSides = NumSidesT;

			math::float2 vertices[NumSides];

			__host__ __device__ ConvexRightPrism() {}

			__host__ __device__ ConvexRightPrism(const math::float2* vertices, unsigned int numSides) 
			{
				if (numSides != NumSides)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("irregular convex right prism num. sides must be exactly %d (numSides=%d)\n", NumSides, numSides);
					asm("trap;");
#else
					throw std::runtime_error(("irregular convex right prism num. sides must be exactly " + std::to_string(NumSides) + " (numSides=" + std::to_string(numSides) + ")").c_str());
#endif
				}
				for (unsigned int i = 0; i < NumSides; i++)
					this->vertices[i] = vertices[i];
			}
			
			__host__ __device__ ConvexRightPrism(const ConvexRightPrism& other) : Shape(other) 
			{
				copyVertices(other.vertices);
			}

			__host__ __device__ __inline__ unsigned int getNumSides() const
			{
				return NumSidesT;
			}

			__host__ __device__ __inline__ math::float2 getVertex(unsigned int i) const
			{
				return vertices[i];
			}

			__host__ __device__ __inline__ ConvexPolygon<NumSidesT, RegularT> newCapFace(bool invert = false) const
			{
				ConvexPolygon<NumSidesT, RegularT> cap;
				cap.invert = invert;
				for (unsigned int i = 0; i < NumSidesT; i++)
					cap.vertices[i] = vertices[i];
				return cap;
			}

			__host__ __device__ __inline__ Quad newSideFace() const
			{
				return{};
			}

			__host__ __device__ __inline__ static ConvexRightPrism<NumSidesT, RegularT> fromPlanarType(const ConvexPolygon<NumSidesT, RegularT>& polygon)
			{
				ConvexRightPrism<NumSidesT, RegularT> prism;
				for (unsigned int i = 0; i < NumSidesT; i++)
					prism.vertices[i] = polygon.vertices[i];
				return prism;
			}

			__host__ __device__ __inline__ ConvexRightPrism& operator=(const ConvexRightPrism& other)
			{
				Shape::operator=(other);
				copyVertices(other.vertices);
				return *this;
			}

			__host__ __inline__ static std::string toString()
			{
				return "ConvexRightPrism<" + std::to_string(NumSidesT) + ", false>";
			}

		};

		template <unsigned int NumSidesT>
		class ConvexRightPrism<NumSidesT, true> : public Shape
		{
		public:
			static_assert(NumSidesT >= 3, "convex right prism cannot have less than 3 side faces");
			static const unsigned int NumSides = NumSidesT;
			bool invert;

			// NOTE: doesn't need to have explicit vertices

			__host__ __device__ ConvexRightPrism() : invert(false) {}
			__host__ __device__ ConvexRightPrism(const ConvexRightPrism& other) : Shape(other), invert(other.invert) {}

			__host__ __device__ __inline__ unsigned int getNumSides() const
			{
				return NumSidesT;
			}

			__host__ __device__ __inline__ math::float2 getVertex(unsigned int i) const
			{
				float sign = 1.0f - 2.0f * invert;
				float angle = 6.2831853071795864769252868f / (float)NumSidesT * i + 1.5707963267948966192313217f;
				return math::float2(cos(angle) * sign, sin(angle) * sign) * 0.5f;
			}

			__host__ __device__ __inline__ ConvexPolygon<NumSidesT, true> newCapFace(bool invert = false) const
			{
				ConvexPolygon<NumSidesT, true> cap;
				cap.invert = invert;
				return cap;
			}

			__host__ __device__ __inline__ Quad newSideFace() const
			{
				return{};
			}

			__host__ __device__ __inline__ static ConvexRightPrism<NumSidesT, true> fromPlanarType(const ConvexPolygon<NumSidesT, true>& polygon)
			{
				return ConvexRightPrism<NumSidesT, true>();
			}

			__host__ __device__ __inline__ ConvexRightPrism& operator=(const ConvexRightPrism& other)
			{
				Shape::operator=(other);
				invert = other.invert;
				return *this;
			}

			__host__ __inline__ static std::string toString()
			{
				return "ConvexRightPrism<" + std::to_string(NumSidesT) + ", true>";
			}

		};

		typedef ConvexRightPrism<3, true> Prism3;
		typedef ConvexRightPrism<4, true> Box;
		typedef ConvexRightPrism<5, true> Prism5;
		typedef ConvexRightPrism<6, true> Prism6;
		typedef ConvexRightPrism<7, true> Prism7;
		typedef ConvexRightPrism<8, true> Prism8;

		template <unsigned int NumSides, bool RegularT>
		struct GetNumFaces < ConvexRightPrism<NumSides, RegularT> >
		{
			static const unsigned int Result = NumSides + 2;

		};

		template <unsigned int NumSides, bool RegularT>
		struct GetCapFaceType < ConvexRightPrism<NumSides, RegularT> >
		{
			typedef ConvexPolygon<NumSides, RegularT> Result;

		};

		template <unsigned int NumSides, bool RegularT>
		struct GetSideFaceType < ConvexRightPrism<NumSides, RegularT> >
		{
			typedef Quad Result;

		};

		template <unsigned int NumSides, bool RegularT>
		struct GetExtrudedType < ConvexPolygon<NumSides, RegularT> >
		{
			typedef ConvexRightPrism<NumSides, RegularT> Result;

		};

		template <unsigned int NumSides>
		struct GetName < ConvexRightPrism<NumSides, false> >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "IrregularConvexRightPrism";
			}

		};

		template <unsigned int NumSides>
		struct GetName < ConvexRightPrism<NumSides, true> >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "RegularTConvexRightPrism";
			}

		};

		template <>
		struct GetName < Box >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Box";
			}

		};

		template <unsigned int NumSides, bool RegularT>
		struct IsPlanar < ConvexRightPrism<NumSides, RegularT> >
		{
			static const bool Result = false;

		};

	}

}
