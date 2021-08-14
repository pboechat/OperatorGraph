#pragma once

#include <cuda_runtime_api.h>
#include <math/matrix.h>
#include <math/vector.h>

namespace PGA
{
	template <typename ShapeT>
	class Symbol : public ShapeT
	{
	public:
		int entryIndex;
		int predecessor;

		__host__ __device__ Symbol() : ShapeT(), entryIndex(-1), predecessor(-1) {}

		__host__ __device__ Symbol(const ShapeT& other) : ShapeT(other), entryIndex(-1), predecessor(-1)
		{
		}

		__host__ __device__ Symbol(const Symbol<ShapeT>& other) : ShapeT(other)
		{
			entryIndex = other.entryIndex;
			predecessor = other.predecessor;
		}

		__host__ __device__ Symbol<ShapeT>& operator=(const Symbol<ShapeT>& other)
		{
			entryIndex = other.entryIndex;
			predecessor = other.predecessor;
			ShapeT::operator=(other);
			return *this;
		}

	};

	template <typename SymbolT>
	struct GetSymbolShape;

	template <typename ShapeT>
	struct GetSymbolShape < Symbol<ShapeT> >
	{
		typedef ShapeT Result;

	};

}
