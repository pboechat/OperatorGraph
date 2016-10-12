#pragma once

#include <string>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "Random.cuh"

namespace PGA
{
	namespace Shapes
	{
		class Shape
		{
		protected:
			math::float3x4 model;
			math::float3 size;
			float seed;
			float customAttribute;

		public:
			__host__ __device__ Shape() :
				model(1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f),
				size(math::float3(1.0f, 1.0f, 1.0f)),
				seed(0.0f), 
				customAttribute(0.0f)
			{
			}

			__host__ __device__ Shape(const Shape& other)
			{
				model = other.model;
				size = other.size;
				seed = other.seed;
				customAttribute = other.customAttribute;
			}

			__host__ __device__ __inline__ math::float3 getPosition() const
			{	
				return math::float3(model._14, model._24, model._34);
			}

			__host__ __device__ __inline__ void setPosition(const math::float3& position)
			{	
				model._14 = position.x; model._24 = position.y; model._34 = position.z;
			}

			__host__ __device__ __inline__ math::float3x3 getRotation() const
			{	
				return math::float3x3(model._11, model._12, model._13,
					model._21, model._22, model._23,
					model._31, model._32, model._33);
			}

			__host__ __device__ __inline__ math::float3 getSize() const
			{
				return size;
			}

			__host__ __device__ __inline__ void setSize(const math::float3& s)
			{
				size = s;
			}

			__host__ __device__ __inline__ math::float3 getHalfExtents() const
			{
				return (size * 0.5f);
			}

			__host__ __device__ __inline__ math::float4x4 getModel4() const
			{
				return math::float4x4(model._11, model._12, model._13, model._14,
					model._21, model._22, model._23, model._24,
					model._31, model._32, model._33, model._34,
					0, 0, 0, 1);
			}

			__host__ __device__ __inline__ math::float3x4 getModel() const
			{
				return model;
			}

			__host__ __device__ __inline__ void setModel(const math::float4x4& m)
			{
				model = math::float3x4(m._11, m._12, m._13, m._14,
					m._21, m._22, m._23, m._24,
					m._31, m._32, m._33, m._34);
			}

			__host__ __device__ __inline__ void setModel(const math::float3x4& m)
			{
				model = m;
			}

			__host__ __device__ __inline__ float getSeed() const
			{
				return seed;
			}

			__host__ __device__ __inline__ float generateNextSeed(float modifier) const
			{
				return Random::nextSeed(seed, static_cast<unsigned int>(modifier));
			}

			__host__ __device__ __inline__ void setSeed(float s)
			{
				seed = s;
			}

			__host__ __device__ __inline__ float operator[](unsigned int i) const
			{
				return at(i);
			}

			__host__ __device__ __inline__ float at(unsigned int i) const
			{
				if (i == 0)
					return model._11;
				else if (i == 1)
					return model._12;
				else if (i == 2)
					return model._13;
				else if (i == 3)
					return model._14;
				else if (i == 4)
					return model._21;
				else if (i == 5)
					return model._22;
				else if (i == 6)
					return model._23;
				else if (i == 7)
					return model._24;
				else if (i == 8)
					return model._31;
				else if (i == 9)
					return model._32;
				else if (i == 10)
					return model._33;
				else if (i == 11)
					return model._34;
				else if (i == 12)
					return size.x;
				else if (i == 13)
					return size.y;
				else if (i == 14)
					return size.z;
				else if (i == 15)
					return seed;
				else if (i == 16)
					return customAttribute;
				else
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
					printf("PGA::Shapes::Shape::at(..): invalid attribute index [i=%d] (CUDA thread %d %d)\n", i, threadIdx.x, blockIdx.x);
					asm("trap;");
#endif
#else
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL == 1 || PGA_INVARIANT_CHECKING_LVL == 3)
					throw std::runtime_error(("PGA::Shapes::Shape::at(..): invalid attribute index [i=" + std::to_string(i) + "]").c_str());
#endif
#endif
				}
				// warning C4715
				return 0;
			}

			__host__ __device__ Shape& operator=(const Shape& other)
			{
				model = other.model;
				size = other.size;
				seed = other.seed;
				customAttribute = other.customAttribute;
				return *this;
			}

			__host__ __device__ __inline__ float getCustomAttribute() const
			{
				return customAttribute;
			}

			__host__ __device__ __inline__ void setCustomAttribute(float value)
			{
				customAttribute = value;
			}

		};

		template <typename Shape>
		struct GetNumFaces;

		template <typename Shape>
		struct GetCapFaceType;

		template <typename Shape>
		struct GetSideFaceType;

		template <typename Shape>
		struct GetExtrudedType;

		template <typename Shape>
		struct GetName;

		template <typename Shape>
		struct IsPlanar;

	}

}
