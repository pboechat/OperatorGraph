#pragma once

#include <cuda_runtime_api.h>

#include <math/vector.h>

#include "GlobalConstants.h"
#include "Triangulation.cuh"
#include "Shapes.cuh"

namespace PGA
{
	//////////////////////////////////////////////////////////////////////////
	template <typename FilterT, unsigned int IndexT>
	struct GenerationFunction;

	//////////////////////////////////////////////////////////////////////////
	template <typename ShapeT, bool ParallelT>
	struct ShapeGenerator;

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int MaxNumVerticesT>
	struct ShapeGenerator < Shapes::DynamicRightPrism<MaxNumVerticesT, true>, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::DynamicRightPrism<MaxNumVerticesT, true>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float3 size = shape.getSize();
			float halfHeight = size.y * 0.5f;
			math::float2x2 capScale = math::float2x2::scale(size.xz());
			math::float4x4 model = shape.getModel4();
			math::float3x3 rotation = shape.getRotation();

			math::float3 normal = rotation * math::float3(0.0f, 1.0f, 0.0f);

			GenFuncT::perVertex(vertexIndex, model * math::float4(0.0f, halfHeight, 0.0f, 1.0f), normal, math::float2(0.5f, 0.5f), terminalIndex, args...);

			unsigned int numSides = shape.getNumSides();
			unsigned int k = vertexIndex + 1;
			for (unsigned int i = 0, j = vertexIndex + numSides; i < numSides; i++, j = k, k++, index += 3)
			{
				const auto& vertex = shape.getVertex(i);
				math::float2 scaledVertex = capScale * vertex;
				GenFuncT::perVertex(k, model * math::float4(scaledVertex.x, halfHeight, -scaledVertex.y, 1.0f), normal, math::float2(0.5f - vertex.x, vertex.y + 0.5f), terminalIndex, args...);
#if defined(PGA_CW)
				GenFuncT::perTriangle(index, vertexIndex, j, k, terminalIndex, args...);
#else
				GenFuncT::perTriangle(index, vertexIndex, k, j, terminalIndex, args...);
#endif
			}

			vertexIndex = k;
			normal = rotation * math::float3(0.0f, -1.0f, 0.0f);

			GenFuncT::perVertex(vertexIndex, model * math::float4(0.0f, -halfHeight, 0.0f, 1.0f), normal, math::float2(0.5f, 0.5f), terminalIndex, args...);

			k = vertexIndex + 1;
			for (unsigned int i = 0, j = vertexIndex + numSides; i < numSides; i++, j = k, k++, index += 3)
			{
				const auto& vertex = shape.getVertex(i);
				math::float2 scaledVertex = capScale * vertex;
				GenFuncT::perVertex(k, model * math::float4(scaledVertex.x, -halfHeight, -scaledVertex.y, 1.0f), normal, math::float2(0.5f - vertex.x, 0.5f - vertex.y), terminalIndex, args...);
#if defined(PGA_CW)
				GenFuncT::perTriangle(index, vertexIndex, k, j, terminalIndex, args...);
#else
				GenFuncT::perTriangle(index, vertexIndex, j, k, terminalIndex, args...);
#endif
			}
			vertexIndex = k;

			for (unsigned int i = numSides - 1, j = 0; j < numSides; i = j, j++, vertexIndex += 4, index += 6)
			{
				math::float2 scaledVertex0 = capScale * shape.getVertex(i);
				math::float2 scaledVertex1 = capScale * shape.getVertex(j);
#if defined(PGA_CW)
				auto v0 = model * math::float4(scaledVertex1.x, halfHeight, -scaledVertex1.y, 1.0f);
				auto v1 = model * math::float4(scaledVertex0.x, halfHeight, -scaledVertex0.y, 1.0f);
				auto v2 = model * math::float4(scaledVertex0.x, -halfHeight, -scaledVertex0.y, 1.0f);
				auto v3 = model * math::float4(scaledVertex1.x, -halfHeight, -scaledVertex1.y, 1.0f);
				normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
				math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
#else
				auto v0 = model * math::float4(scaledVertex0.x, halfHeight, -scaledVertex0.y, 1.0f);
				auto v1 = model * math::float4(scaledVertex1.x, halfHeight, -scaledVertex1.y, 1.0f);
				auto v2 = model * math::float4(scaledVertex1.x, -halfHeight, -scaledVertex1.y, 1.0f);
				auto v3 = model * math::float4(scaledVertex0.x, -halfHeight, -scaledVertex0.y, 1.0f);
				normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
				math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
#endif
				GenFuncT::perVertex(vertexIndex, v0, normal, uv0, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 1, v1, normal, uv1, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 2, v2, normal, uv2, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 3, v3, normal, uv3, terminalIndex, args...);
				GenFuncT::perTriangle(index, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
				GenFuncT::perTriangle(index + 3, vertexIndex, vertexIndex + 2, vertexIndex + 3, terminalIndex, args...);
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int MaxNumVerticesT>
	struct ShapeGenerator < Shapes::DynamicRightPrism<MaxNumVerticesT, false>, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::DynamicRightPrism<MaxNumVerticesT, false>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int indices[(MaxNumVerticesT - 2) * 3];
			unsigned int numVertices = shape.getNumSides();
			// DEBUG:
			if (!Triangulation::template earClipping<MaxNumVerticesT>(shape.vertices, numVertices, indices))
				return;

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float3 size = shape.getSize();
			float halfHeight = size.y * 0.5f;
			math::float2x2 capScale = math::float2x2::scale(shape.getSize().xz());
			math::float4x4 model = shape.getModel4();
			auto rotation = shape.getRotation();
			math::float3 normal1 = rotation * math::float3(0.0f, 1.0f, 0.0f), normal2 = rotation * math::float3(0.0f, -1.0f, 0.0f);

			unsigned int offset1 = vertexIndex, offset2 = vertexIndex + numVertices;
			for (unsigned int i = 0; i < numVertices; i++)
			{
				const auto& vertex = shape.vertices[i];
				math::float2 scaledVertex = capScale * vertex;
				GenFuncT::perVertex(offset1 + i, model * math::float4(scaledVertex.x, halfHeight, -scaledVertex.y, 1.0f), normal1, math::float2(0.5f - vertex.x, vertex.y + 0.5f), terminalIndex, args...);
				GenFuncT::perVertex(offset2 + i, model * math::float4(scaledVertex.x, -halfHeight, -scaledVertex.y, 1.0f), normal2, math::float2(0.5f - vertex.x, 0.5f - vertex.y), terminalIndex, args...);
			}

			unsigned int numTriangles = (numVertices - 2);
			unsigned int offset3 = numTriangles * 3;
			for (unsigned int i = 0, j = 0; i < numTriangles; i++, j += 3)
			{
#if defined(PGA_CW)
				GenFuncT::perTriangle(index + j, offset1 + indices[j], offset1 + indices[j + 1], offset1 + indices[j + 2], terminalIndex, args...);
				GenFuncT::perTriangle(index + offset3 + j, offset2 + indices[j + 2], offset2 + indices[j + 1], offset2 + indices[j], terminalIndex, args...);
#else
				GenFuncT::perTriangle(index + j, offset1 + indices[j + 2], offset1 + indices[j + 1], offset1 + indices[j], terminalIndex, args...);
				GenFuncT::perTriangle(index + offset3 + j, offset2 + indices[j], offset2 + indices[j + 1], offset2 + indices[j + 2], terminalIndex, args...);
#endif
			}

			index += numTriangles * 6;
			vertexIndex += numVertices * 2;
			for (unsigned int i = numVertices - 1, j = 0; j < numVertices; i = j, j++, vertexIndex += 4, index += 6)
			{
				const auto& scaledVertex0 = capScale * shape.vertices[i];
				const auto& scaledVertex1 = capScale * shape.vertices[j];
#if defined(PGA_CW)
				auto v0 = model * math::float4(scaledVertex1.x, halfHeight, -scaledVertex1.y, 1.0f);
				auto v1 = model * math::float4(scaledVertex0.x, halfHeight, -scaledVertex0.y, 1.0f);
				auto v2 = model * math::float4(scaledVertex0.x, -halfHeight, -scaledVertex0.y, 1.0f);
				auto v3 = model * math::float4(scaledVertex1.x, -halfHeight, -scaledVertex1.y, 1.0f);
				auto normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
				math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
#else
				auto v0 = model * math::float4(scaledVertex0.x, halfHeight, -scaledVertex0.y, 1.0f);
				auto v1 = model * math::float4(scaledVertex1.x, halfHeight, -scaledVertex1.y, 1.0f);
				auto v2 = model * math::float4(scaledVertex1.x, -halfHeight, -scaledVertex1.y, 1.0f);
				auto v3 = model * math::float4(scaledVertex0.x, -halfHeight, -scaledVertex0.y, 1.0f);
				auto normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
				math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
#endif
				GenFuncT::perVertex(vertexIndex, v0, normal, uv0, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 1, v1, normal, uv1, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 2, v2, normal, uv2, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 3, v3, normal, uv3, terminalIndex, args...);
				GenFuncT::perTriangle(index, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
				GenFuncT::perTriangle(index + 3, vertexIndex, vertexIndex + 2, vertexIndex + 3, terminalIndex, args...);
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int NumSidesT, bool RegularT>
	struct ShapeGenerator < Shapes::ConvexRightPrism<NumSidesT, RegularT>, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::ConvexRightPrism<NumSidesT, RegularT>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float3 size = shape.getSize();
			float halfHeight = size.y * 0.5f;
			math::float2x2 capScale = math::float2x2::scale(size.xz());
			math::float4x4 model = shape.getModel4();
			math::float3x3 rotation = shape.getRotation();

			math::float3 normal = rotation * math::float3(0.0f, 1.0f, 0.0f);
			GenFuncT::perVertex(vertexIndex, model * math::float4(0.0f, halfHeight, 0.0f, 1.0f), normal, math::float2(0.5f, 0.5f), terminalIndex, args...);

			unsigned int k = vertexIndex + 1;
			for (unsigned int i = 0, j = vertexIndex + NumSidesT; i < NumSidesT; i++, j = k, k++, index += 3)
			{
				const auto& vertex = shape.getVertex(i);
				math::float2 scaledVertex = capScale * vertex;
				GenFuncT::perVertex(k, model * math::float4(scaledVertex.x, halfHeight, -scaledVertex.y, 1.0f), normal, math::float2(0.5f - vertex.x, vertex.y + 0.5f), terminalIndex, args...);
#if defined(PGA_CW)
				GenFuncT::perTriangle(index, vertexIndex, j, k, terminalIndex, args...);
#else
				GenFuncT::perTriangle(index, vertexIndex, k, j, terminalIndex, args...);
#endif
			}

			vertexIndex = k;
			normal = rotation * math::float3(0.0f, -1.0f, 0.0f);
			GenFuncT::perVertex(vertexIndex, model * math::float4(0.0f, -halfHeight, 0.0f, 1.0f), normal, math::float2(0.5f, 0.5f), terminalIndex, args...);

			k = vertexIndex + 1;
			for (unsigned int i = 0, j = vertexIndex + NumSidesT; i < NumSidesT; i++, j = k, k++, index += 3)
			{
				const auto& vertex = shape.getVertex(i);
				math::float2 scaledVertex = capScale * vertex;
				GenFuncT::perVertex(k, model * math::float4(scaledVertex.x, -halfHeight, -scaledVertex.y, 1.0f), normal, math::float2(0.5f - vertex.x, 0.5f - vertex.y), terminalIndex, args...);
#if defined(PGA_CW)
				GenFuncT::perTriangle(index, vertexIndex, k, j, terminalIndex, args...);
#else
				GenFuncT::perTriangle(index, vertexIndex, j, k, terminalIndex, args...);
#endif
			}

			vertexIndex = k;
			for (unsigned int i = NumSidesT - 1, j = 0; j < NumSidesT; i = j, j++, vertexIndex += 4, index += 6)
			{
				math::float2 scaledVertex0 = capScale * shape.getVertex(i);
				math::float2 scaledVertex1 = capScale * shape.getVertex(j);

#if defined(PGA_CW)
				auto v0 = model * math::float4(scaledVertex1.x, halfHeight, -scaledVertex1.y, 1.0f);
				auto v1 = model * math::float4(scaledVertex0.x, halfHeight, -scaledVertex0.y, 1.0f);
				auto v2 = model * math::float4(scaledVertex0.x, -halfHeight, -scaledVertex0.y, 1.0f);
				auto v3 = model * math::float4(scaledVertex1.x, -halfHeight, -scaledVertex1.y, 1.0f);
				normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
				math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
#else
				auto v0 = model * math::float4(scaledVertex0.x, halfHeight, -scaledVertex0.y, 1.0f);
				auto v1 = model * math::float4(scaledVertex1.x, halfHeight, -scaledVertex1.y, 1.0f);
				auto v2 = model * math::float4(scaledVertex1.x, -halfHeight, -scaledVertex1.y, 1.0f);
				auto v3 = model * math::float4(scaledVertex0.x, -halfHeight, -scaledVertex0.y, 1.0f);
				normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
				math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
#endif
				GenFuncT::perVertex(vertexIndex, v0, normal, uv0, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 1, v1, normal, uv1, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 2, v2, normal, uv2, terminalIndex, args...);
				GenFuncT::perVertex(vertexIndex + 3, v3, normal, uv3, terminalIndex, args...);

				GenFuncT::perTriangle(index, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
				GenFuncT::perTriangle(index + 3, vertexIndex, vertexIndex + 2, vertexIndex + 3, terminalIndex, args...);
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int NumSidesT, bool RegularT>
	struct ShapeGenerator < Shapes::ConvexRightPrism<NumSidesT, RegularT>, true >
	{
		static const unsigned int NumThreads = 8;

		template <typename GenFuncT, typename ContextT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(int threadId, Shapes::ConvexRightPrism<NumSidesT, RegularT>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			unsigned int vertexIndex;
			unsigned int index;
			if (threadId == 0)
			{
				GenFuncT::perShape(shape, terminalIndex, args...);
				vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
				index = GenFuncT::allocateIndices(shape, terminalIndex, args...);
			}

			vertexIndex = ContextT::shfl(vertexIndex, 0);
			index = ContextT::shfl(index, 0);

			math::float3 size = shape.getSize();
			math::float4x4 model = shape.getModel4();
			math::float3x3 rotation = shape.getRotation();
			float halfHeight = size.y * 0.5f;

			math::float3 up = rotation * math::float3(0.0f, 1.0f, 0.0f);
			math::float3 down = rotation * math::float3(0.0f, -1.0f, 0.0f);

			unsigned int vi0 = vertexIndex;
			unsigned int vi1 = vertexIndex + 1;
			if (threadId == 0)
			{
				GenFuncT::perVertex(vi0, model * math::float4(0.0f, halfHeight, 0.0f, 1.0f), up, math::float2(0.5f, 0.5f), terminalIndex, args...);
				GenFuncT::perVertex(vi1, model * math::float4(0.0f, -halfHeight, 0.0f, 1.0f), down, math::float2(0.5f, 0.5f), terminalIndex, args...);
			}

			const unsigned int VStride1 = NumSidesT;
			const unsigned int VStride2 = NumSidesT * 2;
			const unsigned int VStride3 = NumSidesT * 3;
			const unsigned int TStride1 = NumSidesT * 3;
			const unsigned int TStride2 = NumSidesT * 6;
			const unsigned int TStride3 = NumSidesT * 9;

			if (threadId < NumThreads)
			{
				unsigned int offset1 = vertexIndex + 2;
				unsigned int offset2 = offset1 + NumSidesT;
				unsigned int offset3 = vertexIndex + 2 * (NumSidesT + 1);
				for (unsigned int i = 0, j = threadId, k = threadId * 3; i < (NumSidesT + 7) / 8; i++, j += 8, k += 24)
				{
					if (j >= NumSidesT) break;

					unsigned int vi2 = offset1 + j;
					unsigned int vi3 = offset2 + j;

					math::float2x2 capScale = math::float2x2::scale(size.xz());
					math::float2 v0 = shape.getVertex(j);
					math::float2 sv0 = capScale * v0;
					unsigned int l = (j + 1) % NumSidesT;
					math::float2 v1 = shape.getVertex(l);
					math::float2 sv1 = capScale * v1;
					
					GenFuncT::perVertex(vi2, model * math::float4(sv0.x, halfHeight, -sv0.y, 1.0f), up, math::float2(0.5f - v0.x, v0.y + 0.5f), terminalIndex, args...);
					GenFuncT::perVertex(vi3, model * math::float4(sv1.x, -halfHeight, -sv1.y, 1.0f), down, math::float2(0.5f - v1.x, 0.5f - v1.y), terminalIndex, args...);

					unsigned int vi4 = offset3 + j;
					unsigned int vi5 = offset3 + j + VStride1;
					unsigned int vi6 = offset3 + j + VStride2;
					unsigned int vi7 = offset3 + j + VStride3;

#if defined(PGA_CW)
					auto v2 = model * math::float4(sv1.x, halfHeight, -sv1.y, 1.0f);
					auto v3 = model * math::float4(sv0.x, halfHeight, -sv0.y, 1.0f);
					auto v4 = model * math::float4(sv0.x, -halfHeight, -sv0.y, 1.0f);
					auto v5 = model * math::float4(sv1.x, -halfHeight, -sv1.y, 1.0f);
					auto normal = normalize(cross((v3 - v2).xyz(), (v4 - v2).xyz()));
					math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
#else
					auto v2 = model * math::float4(sv0.x, halfHeight, -sv0.y, 1.0f);
					auto v3 = model * math::float4(sv1.x, halfHeight, -sv1.y, 1.0f);
					auto v4 = model * math::float4(sv1.x, -halfHeight, -sv1.y, 1.0f);
					auto v5 = model * math::float4(sv0.x, -halfHeight, -sv0.y, 1.0f);
					auto normal = normalize(cross((v4 - v2).xyz(), (v3 - v2).xyz()));
					math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
#endif

					GenFuncT::perVertex(vi4, v2, normal, uv0, terminalIndex, args...);
					GenFuncT::perVertex(vi5, v3, normal, uv1, terminalIndex, args...);
					GenFuncT::perVertex(vi6, v4, normal, uv2, terminalIndex, args...);
					GenFuncT::perVertex(vi7, v5, normal, uv3, terminalIndex, args...);

					unsigned int ti0 = index + k;
					unsigned int ti1 = index + k + TStride1;
					unsigned int ti2 = index + k + TStride2;
					unsigned int ti3 = index + k + TStride3;

					// caps
#if defined(PGA_CW)
					GenFuncT::perTriangle(ti0, vi0, vi2, offset1 + l, terminalIndex, args...);
					GenFuncT::perTriangle(ti1, vi1, offset2 + l, vi3, terminalIndex, args...);
#else
					GenFuncT::perTriangle(ti0, offset1 + l, vi2, vi0, terminalIndex, args...);
					GenFuncT::perTriangle(ti1, vi3, offset2 + l, vi1, terminalIndex, args...);
#endif
					// sides
					GenFuncT::perTriangle(ti2, vi4, vi5, vi6, terminalIndex, args...);
					GenFuncT::perTriangle(ti3, vi4, vi6, vi7, terminalIndex, args...);
				}
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <>
	struct ShapeGenerator < Shapes::Box, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::Box& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int baseIndex = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float4x4 model = shape.getModel4();
			math::float3 halfExtents = shape.getHalfExtents();

#if defined(PGA_CW)
			math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
#else
			math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
#endif

			// top
#if defined(PGA_CW)
			auto v0 = model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			auto v1 = model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			auto v2 = model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			auto v3 = model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			auto normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			auto v0 = model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			auto v1 = model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			auto v2 = model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			auto v3 = model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			auto normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif
			GenFuncT::perVertex(vertexIndex + 0, v0, normal, uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 1, v1, normal, uv1, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 2, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 3, v3, normal, uv3, terminalIndex, args...);
			// bottom
#if defined(PGA_CW)
			v0 = model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v1 = model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v2 = model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			v3 = model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			v0 = model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v1 = model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v2 = model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			v3 = model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif
			GenFuncT::perVertex(vertexIndex + 4, v0, normal, uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 5, v1, normal, uv1, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 6, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 7, v3, normal, uv3, terminalIndex, args...);
			// left
#if defined(PGA_CW)
			v0 = model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v1 = model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v2 = model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			v3 = model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			v0 = model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v1 = model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v2 = model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v3 = model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif
			GenFuncT::perVertex(vertexIndex + 8, v0, normal,  uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 9, v1, normal,  uv1, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 10, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 11, v3, normal, uv3, terminalIndex, args...);
			// right
#if defined(PGA_CW)
			v0 = model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v1 = model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v2 = model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v3 = model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			v0 = model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v1 = model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v2 = model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			v3 = model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif
			GenFuncT::perVertex(vertexIndex + 12, v0, normal, uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 13, v1, normal, uv1, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 14, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 15, v3, normal, uv3, terminalIndex, args...);
			// front
#if defined(PGA_CW)
			v0 = model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v1 = model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v2 = model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v3 = model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			v0 = model * math::float4(-halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v1 = model * math::float4(halfExtents.x, halfExtents.y, halfExtents.z, 1.0f);
			v2 = model * math::float4(halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			v3 = model * math::float4(-halfExtents.x, -halfExtents.y, halfExtents.z, 1.0f);
			normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif
			GenFuncT::perVertex(vertexIndex + 16, v0, normal, uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 17, v1, normal, uv1,	terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 18, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 19, v3, normal, uv3, terminalIndex, args...);
			// back
#if defined(PGA_CW)
			v0 = model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v1 = model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v2 = model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			v3 = model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			v0 = model * math::float4(halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v1 = model * math::float4(-halfExtents.x, halfExtents.y, -halfExtents.z, 1.0f);
			v2 = model * math::float4(-halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			v3 = model * math::float4(halfExtents.x, -halfExtents.y, -halfExtents.z, 1.0f);
			normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif
			GenFuncT::perVertex(vertexIndex + 20, v0, normal, uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 21, v1, normal, uv1, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 22, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 23, v3, normal, uv3, terminalIndex, args...);

			for (unsigned int faceIndex = 0; faceIndex < 6; faceIndex++, baseIndex += 6, vertexIndex += 4)
			{
				GenFuncT::perTriangle(baseIndex, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
				GenFuncT::perTriangle(baseIndex + 3, vertexIndex, vertexIndex + 2, vertexIndex + 3, terminalIndex, args...);
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <>
	struct ShapeGenerator < Shapes::Box, true >
	{
		static const unsigned int NumThreads = 12;

		// NOTES:
		//
		// - THREAD x FACE:
		//
		// 0 = BACK
		// 1 = FRONT
		// 2 = BACK
		// 3 = FRONT
		// 4 = LEFT
		// 5 = RIGHT
		// 6 = LEFT
		// 7 = RIGHT
		// 8 = BOTTOM
		// 9 = TOP
		// 10 = BOTTOM
		// 11 = TOP
		//
		// - THREAD x (VERTEX INDEX & COORDS.):
		// 0 = 1 -> (1, 1, -1), 0 -> (-1, 1, -1) *BACK*
		// 1 = 5 -> (-1, 1, 1), 4 -> (1, 1, 1) *FRONT*
		// 2 = 3 -> (-1, -1, -1), 2 -> (1, -1, -1) *BACK*
		// 3 = 7 -> (1, -1, 1), 6 -> (-1, -1, 1) *FRONT*
		// 4 = 9 -> (-1, 1, -1), 8 -> (-1, 1, 1) *LEFT*
		// 5 = 13 -> (1, 1, 1), 12 -> (1, 1, -1) *RIGHT*
		// 6 = 11 -> (-1, -1, 1), 10 -> (-1, -1, -1) *LEFT*
		// 7 = 15 -> (1, -1, -1), 14 -> (1, -1, 1) *RIGHT*
		// 8 = 17 -> (-1, -1, 1), 16 -> (1, -1, 1) *BOTTOM*
		// 9 = 21 -> (-1, 1, -1), 20 -> (1, 1, -1) *TOP*
		// 10 = 19 -> (1, -1, -1), 18 -> (-1, -1, -1) *BOTTOM*
		// 11 = 23 -> (1, 1, 1), 22 -> (-1, 1, 1) *TOP*
		//
		// - THREAD x (VERTEX INDEX & INDEX):
		// 0 = 0 -> 0, 1 -> 12
		// 1 = 4 -> 1, 5 -> 13
		// 2 = 2 -> 2, 3 -> 14
		// 3 = 6 -> 3, 7 -> 15
		// 4 = 8 -> 4, 9 -> 16
		// 5 = 12 -> 5, 13 -> 17
		// 6 = 10 -> 6, 11 -> 18
		// 7 = 14 -> 7, 15 -> 19
		// 8 = 16 -> 8, 17 -> 20
		// 9 = 20 -> 9, 21 -> 21
		// 10 = 18 -> 10, 19 -> 22
		// 11 = 22 -> 11, 23 -> 23
		//
		// - THREAD x TRIANGLE:
		// 0 = 0, 12, 2
		// 1 = 1, 13, 3
		// 2 = 0, 2, 14
		// 3 = 1, 3, 15
		// 4 = 4, 16, 5
		// 5 = 4, 5, 17
		// ...

		template <typename GenFuncT, typename ContextT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(int threadId, Shapes::Box& shape, unsigned int terminalIndex, ArgsT... args)
		{
			unsigned int vertexIndex;
			unsigned int index;
			if (threadId == 0)
			{
				GenFuncT::perShape(shape, terminalIndex, args...);
				vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
				index = GenFuncT::allocateIndices(shape, terminalIndex, args...);
			}

			vertexIndex = ContextT::shfl(vertexIndex, 0);
			index = ContextT::shfl(index, 0);

			if (threadId < NumThreads)
			{
				math::float4x4 model = shape.getModel4() * math::float4x4::scale(shape.getHalfExtents());

				float sign = ((threadId & 0x1) * 2.0f) - 1.0f;
				float f0 = (float)(threadId < 4);
				float f2 = (threadId & 0x4) * 0.25f;
				float f3 = (threadId & 0x8) * 0.125f;
				int b0 = threadId & 0x1;
				int b1 = (threadId & 0x2) != 0;

				int baseVertexIndex = vertexIndex + threadId;
				int nextBaseVertexIndex = baseVertexIndex + 12;
				int adjustment = (b0 ^ b1) * sign;
				int baseIndex = index + (threadId * 3) + (adjustment * 3);

				math::float3 z(f2 * sign, f3 * sign, f0 * sign);
				math::float3 y(0.0f, 1.0f - f3, f3 * sign);
				math::float3 x = cross(y, z);

				float sign2 = (threadId & 0x2) - 1.0f;
#if defined(PGA_CW)
				auto v1 = z + (y * -sign2) + (x * -sign2);
				auto v2 = z + (y * -sign2) + (x * sign2);
#else
				auto v1 = z + (y * -sign2) + (x * sign2);
				auto v2 = z + (y * -sign2) + (x * -sign2);
#endif

				math::float3 normal = normalize(shape.getRotation() * z);

				int b3 = (threadId & 0x8) != 0;

#if defined(PGA_CW)
				float s1 = b3 ^ b1;
				float t1 = b3 ^ !b1;
				float s2 = b3 ^ !b1;
				float t2 = b3 ^ !b1;
#else
				float s1 = b3 ^ !b1;
				float t1 = b3 ^ !b1;
				float s2 = b3 ^ b1;
				float t2 = b3 ^ !b1;
#endif

				GenFuncT::perVertex(baseVertexIndex, model * math::float4(v1, 1.0f), normal, math::float2(s1, t1), terminalIndex, args...);
				GenFuncT::perVertex(nextBaseVertexIndex, model * math::float4(v2, 1.0f), normal, math::float2(s2, t2), terminalIndex, args...);

				int b1_c = 1 - b1;
				GenFuncT::perTriangle(baseIndex, baseVertexIndex - (b1 * 2), baseVertexIndex + (12 * b1_c), baseVertexIndex + 2 + (10 * b1), terminalIndex, args...);
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int NumSidesT, bool RegularT>
	struct ShapeGenerator < Shapes::ConvexPolygon<NumSidesT, RegularT>, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::ConvexPolygon<NumSidesT, RegularT>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float4x4 model = shape.getModel4();
			math::float2x2 scale = math::float2x2::scale(shape.getSize().xy());
			math::float3 normal = shape.getRotation() * math::float3(0.0f, 0.0f, 1.0f);

			GenFuncT::perVertex(vertexIndex, math::float4(model.column4().xyz(), 1.0f), normal, math::float2(0.5f, 0.5f), terminalIndex, args...);

			for (unsigned int i = 0, j = vertexIndex + NumSidesT, k = vertexIndex + 1; i < NumSidesT; i++, j = k, k++, index += 3)
			{
				const auto& vertex = shape.getVertex(i);
				math::float2 scaledVertex = scale * vertex;
				GenFuncT::perVertex(k, model * math::float4(scaledVertex, 0.0f, 1.0f), normal, math::float2(0.5f - vertex.x, vertex.y + 0.5f), terminalIndex, args...);
#if defined(PGA_CW)
				GenFuncT::perTriangle(index, vertexIndex, j, k, terminalIndex, args...);
#else
				GenFuncT::perTriangle(index, vertexIndex, k, j, terminalIndex, args...);
#endif
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <>
	struct ShapeGenerator < Shapes::Quad, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::Quad& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float4x4 model = shape.getModel4();
			math::float3 halfExtents = shape.getHalfExtents();

#if defined(PGA_CW)
			auto v0 = model * math::float4(halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v1 = model * math::float4(-halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v2 = model * math::float4(-halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			auto v3 = model * math::float4(halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
			math::float3 normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			auto v0 = model * math::float4(-halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v1 = model * math::float4(halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v2 = model * math::float4(halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			auto v3 = model * math::float4(-halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
			math::float3 normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif

			GenFuncT::perVertex(vertexIndex, v0, normal, uv0, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 1, v1, normal, uv1, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 2, v2, normal, uv2, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 3, v3, normal, uv3, terminalIndex, args...);

			GenFuncT::perTriangle(index, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
			GenFuncT::perTriangle(index + 3, vertexIndex, vertexIndex + 2, vertexIndex + 3, terminalIndex, args...);
		}

		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::Quad& shape, unsigned int terminalIndex, const math::float4& texCoords, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float4x4 model = shape.getModel4();
			math::float3 halfExtents = shape.getHalfExtents();

#if defined(PGA_CW)
			auto v0 = model * math::float4(halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v1 = model * math::float4(-halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v2 = model * math::float4(-halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			auto v3 = model * math::float4(halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			math::float2 uv0(0.0f, 1.0f), uv1(1.0f, 1.0f), uv2(1.0f, 0.0f), uv3(0.0f, 0.0f);
			math::float3 normal = normalize(cross((v2 - v1).xyz(), (v0 - v1).xyz()));
#else
			auto v0 = model * math::float4(-halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v1 = model * math::float4(halfExtents.x, halfExtents.y, 0.0f, 1.0f);
			auto v2 = model * math::float4(halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			auto v3 = model * math::float4(-halfExtents.x, -halfExtents.y, 0.0f, 1.0f);
			math::float2 uv0(1.0f, 1.0f), uv1(0.0f, 1.0f), uv2(0.0f, 0.0f), uv3(1.0f, 0.0f);
			math::float3 normal = normalize(cross((v0 - v1).xyz(), (v2 - v1).xyz()));
#endif

			GenFuncT::perVertex(vertexIndex, v0, normal, math::float2(texCoords.y, texCoords.z), terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 1, v1, normal, math::float2(texCoords.y, texCoords.w), terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 2, v2, normal, math::float2(texCoords.x, texCoords.w), terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 3, v3, normal, math::float2(texCoords.x, texCoords.z), terminalIndex, args...);

			GenFuncT::perTriangle(index, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
			GenFuncT::perTriangle(index + 3, vertexIndex, vertexIndex + 2, vertexIndex + 3, terminalIndex, args...);
		}

	};
	
	//////////////////////////////////////////////////////////////////////////
	template <bool RegularT>
	struct ShapeGenerator < Shapes::ConvexPolygon<3, RegularT>, false >
	{
		static const unsigned int NumThreads = 1;

		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::ConvexPolygon<3, RegularT>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float4x4 model = shape.getModel4();
			math::float2x2 scale = math::float2x2::scale(shape.getSize().xy());
			math::float3 normal = shape.getRotation() * math::float3(0.0f, 0.0f, 1.0f);

			const auto& v0 = shape.getVertex(0);
			const auto& v1 = shape.getVertex(1);
			const auto& v2 = shape.getVertex(2);

			math::float2 scaledVertex0 = scale * v0;
			math::float2 scaledVertex1 = scale * v1;
			math::float2 scaledVertex2 = scale * v2;

			GenFuncT::perVertex(vertexIndex, model * math::float4(v0, 0.0f, 1.0f), normal, v0 + 0.5f, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 1, model * math::float4(v1, 0.0f, 1.0f), normal, v1 + 0.5f, terminalIndex, args...);
			GenFuncT::perVertex(vertexIndex + 2, model * math::float4(v2, 0.0f, 1.0f), normal, v2 + 0.5f, terminalIndex, args...);
#if defined(PGA_CW)
			GenFuncT::perTriangle(index, vertexIndex + 2, vertexIndex + 1, vertexIndex, terminalIndex, args...);
#else
			GenFuncT::perTriangle(index, vertexIndex, vertexIndex + 1, vertexIndex + 2, terminalIndex, args...);
#endif
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int MaxNumVerticesT>
	struct ShapeGenerator < Shapes::DynamicPolygon<MaxNumVerticesT, true>, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::DynamicPolygon<MaxNumVerticesT, true>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			math::float4x4 model = shape.getModel4();
			math::float2x2 scale = math::float2x2::scale(shape.getSize().xy());
			math::float3 normal = shape.getRotation() * math::float3(0.0f, 0.0f, 1.0f);

			GenFuncT::perVertex(vertexIndex, math::float4(model.column4().xyz(), 1.0f), normal, math::float2(0.5f, 0.5f), terminalIndex, args...);

			unsigned int numSides = shape.numSides;
			for (unsigned int i = 0, j = vertexIndex + numSides, k = vertexIndex + 1; i < numSides; i++, j = k, k++, index += 3)
			{
				const auto& vertex = shape.getVertex(i);
				math::float2 scaledVertex = scale * vertex;
				GenFuncT::perVertex(k, model * math::float4(scaledVertex, 0.0f, 1.0f), normal, math::float2(0.5f - vertex.x, vertex.y + 0.5f), terminalIndex, args...);
#if defined(PGA_CW)
				GenFuncT::perTriangle(index, vertexIndex, j, k, terminalIndex, args...);
#else
				GenFuncT::perTriangle(index, vertexIndex, k, j, terminalIndex, args...);
#endif
			}
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <unsigned int MaxNumVerticesT>
	struct ShapeGenerator < Shapes::DynamicPolygon<MaxNumVerticesT, false>, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::DynamicPolygon<MaxNumVerticesT, false>& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);

			unsigned int indices[(MaxNumVerticesT - 2) * 3];
			unsigned int numVertices = shape.getNumSides();
			// DEBUG:
			if (!Triangulation::template earClipping<MaxNumVerticesT>(shape.vertices, numVertices, indices))
				return;

			math::float4x4 model = shape.getModel4();
			math::float2x2 scale = math::float2x2::scale(shape.getSize().xy());
			math::float3 normal = shape.getRotation() * math::float3(0.0f, 0.0f, 1.0f);

			float sign = 1.0f - 2.0f * shape.invert;
			for (unsigned int i = 0; i < numVertices; i++)
			{
				const auto& vertex = shape.vertices[i];
				math::float2 scaledVertex = scale * vertex;
				GenFuncT::perVertex(vertexIndex + i, model * math::float4(scaledVertex.x, scaledVertex.y * sign, 0.0f, 1.0f), normal, math::float2(0.5f - vertex.x, vertex.y + 0.5f), terminalIndex, args...);
			}

			unsigned int numTriangles = (numVertices - 2);
			for (unsigned int i = 0, j = 0; i < numTriangles; i++, j += 3)
#if defined(PGA_CW)
				GenFuncT::perTriangle(index + j, vertexIndex + indices[j + 2 * shape.invert], vertexIndex + indices[j + 1], vertexIndex + indices[j + 2 * !shape.invert], terminalIndex, args...);
#else
				GenFuncT::perTriangle(index + j, vertexIndex + indices[j + 2 * !shape.invert], vertexIndex + indices[j + 1], vertexIndex + indices[j + 2 * shape.invert], terminalIndex, args...);
#endif
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <>
	struct ShapeGenerator < Shapes::Sphere, false >
	{
		template <typename GenFuncT, typename... ArgsT>
		__host__ __device__ __inline__ static void run(Shapes::Sphere& shape, unsigned int terminalIndex, ArgsT... args)
		{
			GenFuncT::perShape(shape, terminalIndex, args...);

			math::float4 translation = math::float4(shape.getPosition(), 1);

			float thetaStep = 2.0f * math::constants<float>::pi() / (Constants::NumSphereSlices - 1.0f);
			float phiStep = math::constants<float>::pi() / (Constants::NumSphereSlices - 1.0f);
			unsigned int vertexIndex = GenFuncT::allocateVertices(shape, terminalIndex, args...);
			unsigned int index = GenFuncT::allocateIndices(shape, terminalIndex, args...);
			unsigned int currentVertexIndex = vertexIndex;
			float radius = shape.getRadius();
			float theta = 0.0f;
			for (auto i = 0; i < Constants::NumSphereSlices; i++, theta += thetaStep)
			{
				float cosTheta = cos(theta);
				float sinTheta = sin(theta);
				float phi = 0.0f;
				for (auto j = 0; j < Constants::NumSphereSlices; j++, phi += phiStep)
				{
					float nx = -sin(phi) * cosTheta;
					float ny = -cos(phi);
					float nz = -sin(phi) * sinTheta;

					float n = sqrt(nx * nx + ny * ny + nz * nz);
					if (n < 0.99f || n > 1.01f)
					{
						nx = nx / n;
						ny = ny / n;
						nz = nz / n;
					}

					// NOTE: texture repeats twice horizontally
					float tx = theta / math::constants<float>::pi();
					float ty = phi / math::constants<float>::pi();

					GenFuncT::perVertex(currentVertexIndex++, translation + math::float4(nx * radius, ny * radius, nz * radius, 1), math::float3(nx, ny, nz), math::float2(tx, ty), terminalIndex, args...);
				}
			}

			for (auto i = 0; i < Constants::NumSphereSlices - 1; i++)
			{
				for (auto j = 0; j < Constants::NumSphereSlices - 1; j++, index += 6)
				{
					unsigned int baseIndex = vertexIndex + (i * Constants::NumSphereSlices + j);
#if defined(PGA_CW)
					GenFuncT::perTriangle(index, baseIndex, baseIndex + Constants::NumSphereSlices + 1, baseIndex + Constants::NumSphereSlices, terminalIndex, args...);
					GenFuncT::perTriangle(index + 3, baseIndex, baseIndex + 1, baseIndex + Constants::NumSphereSlices + 1, terminalIndex, args...);
#else
					GenFuncT::perTriangle(index, baseIndex, baseIndex + Constants::NumSphereSlices, baseIndex + Constants::NumSphereSlices + 1, terminalIndex, args...);
					GenFuncT::perTriangle(index + 3, baseIndex, baseIndex + Constants::NumSphereSlices + 1, baseIndex + 1, terminalIndex, args...);
#endif
				}
			}
		}

	};

}
