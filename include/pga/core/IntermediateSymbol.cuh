#pragma once

#include <cuda_runtime_api.h>

namespace PGA
{
	namespace ContextSensitivity
	{
		template <typename ShapeT>
		struct IntermediateSymbol : public ShapeT
		{
			int colliderTag;

			__host__ __device__ IntermediateSymbol() : ShapeT(), colliderTag(-1) {}

			__host__ __device__ IntermediateSymbol(const ShapeT& other) : ShapeT(other), colliderTag(-1)
			{
			}

			__host__ __device__ IntermediateSymbol(const IntermediateSymbol<ShapeT>& other) : ShapeT(other)
			{
				colliderTag = other.colliderTag;
			}

			__host__ __device__ IntermediateSymbol<ShapeT>& operator=(const IntermediateSymbol<ShapeT>& other)
			{
				colliderTag = other.colliderTag;
				ShapeT::operator=(other);
				return *this;
			}

		};

	}

}
