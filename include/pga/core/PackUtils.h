#pragma once

#include <math.h>
#include <cuda_runtime_api.h>

#include "OperandType.h"

namespace PGA
{
	class PackUtils
	{
	public:
		static const unsigned int FloatPackingPrecision = 4096;

		PackUtils() = delete;

		static float packOperand(OperandType p, float x)
		{
			unsigned int r = *reinterpret_cast<unsigned int*>(&x);
			unsigned int q = (r >> 2) | (((unsigned int)p) << 30);
			return *reinterpret_cast<float*>(&q);
		}

		static float packFloat2(float x, float y)
		{
			return floorf(x * (FloatPackingPrecision - 1.0f)) * FloatPackingPrecision + floorf(y * (FloatPackingPrecision - 1.0f));
		}

		static __host__ __device__ __inline__ void unpackOperand(float a, OperandType& p, float& x)
		{
			unsigned int r = *reinterpret_cast<unsigned*>(&a);
			unsigned int q = r >> 30;
			p = static_cast<OperandType>(*reinterpret_cast<unsigned int*>(&q));
			r <<= 2;
			x = *reinterpret_cast<float*>(&r);
		}

		static __host__ __device__ __inline__ void unpackFloat2(float a, float& x, float& y)
		{
			y = fmodf(a, (float)FloatPackingPrecision) / (FloatPackingPrecision - 1.0f);
			x = floorf(a / (float)FloatPackingPrecision) / (FloatPackingPrecision - 1.0f);
		}

	};

}
