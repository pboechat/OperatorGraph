#pragma once

#include "Shape.cuh"

#include <cuda_runtime_api.h>

#include <string>

namespace PGA
{
	namespace Shapes
	{
		class Sphere : public Shape
		{
		public:
			__host__ __device__ Sphere() {}
			__host__ __device__ Sphere(const Sphere& other) : Shape(other) {}

			__host__ __device__ __inline__ float getRadius() const
			{
				return size.x;
			}

			__host__ __device__ __inline__ void setRadius(float radius)
			{
				// NOTE: setRadius works just like an uniform scaling
				size.x = size.y = size.z = radius;
			}

			__host__ __device__ __inline__ Sphere& operator=(const Sphere& other)
			{
				Shape::operator=(other);
				return *this;
			}

			__host__ __inline__ static std::string toString()
			{
				return "Sphere";
			}
		};

		template<>
		struct GetName < Sphere >
		{
			__host__ __device__ __inline__ static const char* Result()
			{
				return "Sphere";
			}

		};

		template <>
		struct IsPlanar < Sphere >
		{
			static const bool Result = false;

		};

	}

}
