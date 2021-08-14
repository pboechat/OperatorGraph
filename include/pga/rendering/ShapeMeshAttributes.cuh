#pragma once

#include <cuda_runtime_api.h>
#include <pga/core/GlobalConstants.h>
#include <pga/core/Shapes.cuh>

namespace PGA
{
	namespace Rendering
	{
		template <typename ShapeT>
		struct ShapeMeshAttributes;

		template <>
		struct ShapeMeshAttributes < Shapes::Box >
		{
			static const unsigned int NumVertices = 24;
			static const unsigned int NumIndices = 36;

			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::Box& shape)
			{
				return NumVertices;
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::Box& shape)
			{
				return NumIndices;
			}

		};

		template <>
		struct ShapeMeshAttributes < Shapes::Quad >
		{
			static const unsigned int NumVertices = 4;
			static const unsigned int NumIndices = 6;

			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::Quad& shape)
			{
				return NumVertices;
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::Quad& shape)
			{
				return NumIndices;
			}

		};

		template <>
		struct ShapeMeshAttributes < Shapes::Sphere >
		{
			static const int NumVertices = PGA::Constants::NumSphereSlices * PGA::Constants::NumSphereSlices;
			static const int NumIndices = ((PGA::Constants::NumSphereSlices - 1) * (PGA::Constants::NumSphereSlices - 1) * 6);

			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::Sphere& shape)
			{
				return NumVertices;
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::Sphere& shape)
			{
				return NumIndices;
			}

		};

		template <bool Regular>
		struct ShapeMeshAttributes < Shapes::ConvexPolygon<3, Regular> >
		{
			static const int NumVertices = 3;
			static const int NumIndices = 3;

			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::ConvexPolygon<3, Regular>& shape)
			{
				return NumVertices;
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::ConvexPolygon<3, Regular>& shape)
			{
				return NumIndices;
			}

		};

		template <unsigned int NumSides, bool Regular>
		struct ShapeMeshAttributes < Shapes::ConvexPolygon<NumSides, Regular> >
		{
			static const int NumVertices = (NumSides + 1);
			static const int NumIndices = NumSides * 3;

			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::ConvexPolygon<NumSides, Regular>& shape)
			{
				return NumVertices;
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::ConvexPolygon<NumSides, Regular>& shape)
			{
				return NumIndices;
			}

		};

		template <unsigned int NumSides, bool Regular>
		struct ShapeMeshAttributes < Shapes::ConvexRightPrism<NumSides, Regular> >
		{
			static const int NumVertices = NumSides * 4 + 2 * (NumSides + 1);
			static const int NumIndices = NumSides * 12;

			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::ConvexRightPrism<NumSides, Regular>& shape)
			{
				return NumVertices;
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::ConvexRightPrism<NumSides, Regular>& shape)
			{
				return NumIndices;
			}

		};

		template <unsigned int MaxNumVerticesT, bool ConvexT>
		struct ShapeMeshAttributes < Shapes::DynamicPolygon<MaxNumVerticesT, ConvexT> >
		{
			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::DynamicPolygon<MaxNumVerticesT, ConvexT>& shape)
			{
				return (shape.numSides + 1);
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::DynamicPolygon<MaxNumVerticesT, ConvexT>& shape)
			{
				return (shape.numSides * 3);
			}

		};

		template <unsigned int MaxNumVerticesT>
		struct ShapeMeshAttributes < Shapes::DynamicRightPrism<MaxNumVerticesT, true> >
		{
			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::DynamicRightPrism<MaxNumVerticesT, true>& shape)
			{
				return 4 * shape.numSides + 2 * (shape.numSides + 1);
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::DynamicRightPrism<MaxNumVerticesT, true>& shape)
			{
				return 12 * shape.numSides;
			}

		};

		template <unsigned int MaxNumVerticesT>
		struct ShapeMeshAttributes < Shapes::DynamicRightPrism<MaxNumVerticesT, false> >
		{
			__host__  __device__ __inline__ static unsigned int getNumVertices(const Shapes::DynamicRightPrism<MaxNumVerticesT, false>& shape)
			{
				// NOTE: a triangulation of a simple polygon of N vertices always leads to N - 2 triangles
				return 4 * shape.numSides + 2 * ((shape.numSides - 2) * 3);
			}

			__host__  __device__ __inline__ static unsigned int getNumIndices(const Shapes::DynamicRightPrism<MaxNumVerticesT, false>& shape)
			{
				// TODO:
				return 6 * (shape.numSides + (shape.numSides - 2));
			}

		};

	}

}