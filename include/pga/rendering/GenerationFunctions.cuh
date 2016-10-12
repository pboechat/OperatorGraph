#pragma once

#include <string>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <vector_functions.h>

#include <pga/core/ShapeGenerator.cuh>
#include "TriangleMeshData.h"
#include "InstancedTriangleMeshData.h"
#include "RenderingGlobalVariables.cuh"
#include "ShapeMeshAttributes.cuh"

namespace PGA
{
	namespace Rendering
	{
		struct GenFuncFilter;
	
		__host__ __device__ __inline__ void fastAssign(math::float4& dst, const math::float4& src)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			*((float4*)&dst) = make_float4(src.x, src.y, src.z, src.w);
#else
			dst = src;
#endif
		}

		__host__ __device__ __inline__ void fastAssign(math::float3& dst, const math::float3& src)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			*((float3*)&dst) = make_float3(src.x, src.y, src.z);
#else
			dst = src;
#endif
		}

		__host__ __device__ __inline__ void fastAssign(math::float2& dst, const math::float2& src)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			*((float2*)&dst) = make_float2(src.x, src.y);
#else
			dst = src;
#endif
		}

	}

	template <>
	struct GenerationFunction < Rendering::GenFuncFilter, 0 >
	{
		template <typename ShapeT, typename... ArgsT>
		__host__ __device__ __inline__ static void perShape(ShapeT& shape, unsigned int terminalIndex, ArgsT... args)
		{
		}

		template <typename ShapeT, typename... ArgsT>
		__host__ __device__ __inline__ static unsigned int allocateVertices(ShapeT& shape, unsigned int terminalIndex, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::allocateVertices(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::allocateVertices(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			Rendering::TriangleMeshData& triangleMesh = Rendering::GlobalVars::getTriangleMesh(terminalIndex);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return atomicAdd(&triangleMesh.numVertices, Rendering::ShapeMeshAttributes<ShapeT>::getNumVertices(shape));
#else
			auto i = triangleMesh.numVertices;
			triangleMesh.numVertices += Rendering::ShapeMeshAttributes<ShapeT>::getNumVertices(shape);
			return i;
#endif
		}

		template <typename... ArgsT>
		__host__ __device__ __inline__ static void perVertex(unsigned int vertexIndex, const math::float4& vertex, const math::float3& normal, const math::float2& uv, unsigned int terminalIndex, float attr1, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::perVertex(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::perVertex(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			auto& triangleMesh = Rendering::GlobalVars::getTriangleMesh(terminalIndex);
			unsigned int maxNumVertices = triangleMesh.maxNumVertices;
			unsigned int v = vertexIndex % maxNumVertices;
			auto& vertexAttributes = triangleMesh.verticesAttributes[v];
			Rendering::fastAssign(vertexAttributes.position, vertex);
			Rendering::fastAssign(vertexAttributes.normal, normal);
			Rendering::fastAssign(vertexAttributes.uv, uv);
		}

		template <typename ShapeT, typename... ArgsT>
		__host__ __device__ __inline__ static unsigned int allocateIndices(ShapeT& shape, unsigned int terminalIndex, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::allocateIndices(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::allocateIndices(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			Rendering::TriangleMeshData& triangleMesh = Rendering::GlobalVars::getTriangleMesh(terminalIndex);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return atomicAdd(&triangleMesh.numIndices, Rendering::ShapeMeshAttributes<ShapeT>::getNumIndices(shape));
#else
			auto i = triangleMesh.numIndices;
			triangleMesh.numIndices += Rendering::ShapeMeshAttributes<ShapeT>::getNumIndices(shape);
			return i;
#endif
		}

		template <typename... ArgsT>
		__host__ __device__ __inline__ static void perTriangle(unsigned int index, unsigned int vertex0Index, unsigned int vertex1Index, unsigned int vertex2Index, unsigned int terminalIndex, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::perTriangle(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 0>::perTriangle(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			Rendering::TriangleMeshData& triangleMesh = Rendering::GlobalVars::getTriangleMesh(terminalIndex);
			auto maxNumIndices = triangleMesh.maxNumIndices;
			auto i1 = index % maxNumIndices;
			auto i2 = (index + 1) % maxNumIndices;
			auto i3 = (index + 2) % maxNumIndices;
			triangleMesh.indices[i1] = vertex0Index;
			triangleMesh.indices[i2] = vertex1Index;
			triangleMesh.indices[i3] = vertex2Index;
		}

	};

	template <>
	struct GenerationFunction < Rendering::GenFuncFilter, 1 >
	{
		template <typename ShapeT, typename... ArgsT>
		__host__ __device__ __inline__ static void perShape(ShapeT& shape, unsigned int terminalIndex, float attr1, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumInstancedTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 1>::perShape(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumInstancedTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumInstancedTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 1>::perShape(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumInstancedTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumInstancedTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			auto& instancedTriangleMesh = Rendering::GlobalVars::getInstancedTriangleMesh(terminalIndex);
			auto maxNumInstances = instancedTriangleMesh.maxNumInstances;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			auto i = atomicAdd(&instancedTriangleMesh.numInstances, 1) % maxNumInstances;
#else
			auto i = instancedTriangleMesh.numInstances % maxNumInstances;
			instancedTriangleMesh.numInstances++;
#endif
#ifdef PGA_COLUMN_MAJOR
			auto model = shape.getModel4() * math::float4x4::scale(shape.getSize());
#else
			auto model = transpose(shape.getModel4() * math::float4x4::scale(shape.getSize()));
#endif
			auto& instanceAttribute = instancedTriangleMesh.instancesAttributes[i];
			instanceAttribute.modelMatrix = model;
			instanceAttribute.custom = attr1;
		}

		template <unsigned int MaxNumVerticesT, bool ConvexT, typename... ArgsT>
		__host__ __device__ __inline__ static void perShape(Shapes::DynamicRightPrism<MaxNumVerticesT, ConvexT>& shape, unsigned int terminalIndex, float attr1, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumInstancedTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 1>::perShape(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumInstancedTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumInstancedTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 1>::perShape(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumInstancedTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumInstancedTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			auto& instancedTriangleMesh = Rendering::GlobalVars::getInstancedTriangleMesh(terminalIndex);
			auto maxNumInstances = instancedTriangleMesh.maxNumInstances;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			auto i = atomicAdd(&instancedTriangleMesh.numInstances, 1) % maxNumInstances;
#else
			auto i = instancedTriangleMesh.numInstances % maxNumInstances;
			instancedTriangleMesh.numInstances++;
#endif
#ifdef PGA_COLUMN_MAJOR
			auto model = shape.getAdjustedModel() * math::float4x4::scale(shape.getSize());
#else
			auto model = transpose(shape.getAdjustedModel() * math::float4x4::scale(shape.getSize()));
#endif
			auto& instanceAttribute = instancedTriangleMesh.instancesAttributes[i];
			instanceAttribute.modelMatrix = model;
			instanceAttribute.custom = attr1;
		}

		template <unsigned int MaxNumVerticesT, bool ConvexT, typename... ArgsT>
		__host__ __device__ __inline__ static void perShape(Shapes::DynamicPolygon<MaxNumVerticesT, ConvexT>& shape, unsigned int terminalIndex, float attr1, ArgsT... args)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
			if (terminalIndex >= Rendering::GlobalVars::getNumInstancedTriangleMeshes())
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 1>::perShape(..): out of boundaries buffer access [terminalIndex=%d, PGA::Rendering::GlobalVars::getNumInstancedTriangleMeshes()=%d] (CUDA thread %d %d)\n", terminalIndex, Rendering::GlobalVars::getNumInstancedTriangleMeshes(), threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Rendering::GenerationFunction<PGA::Rendering::GenFuncFilter, 1>::perShape(..): out of boundaries buffer access [terminalIndex=" + std::to_string(terminalIndex) + ", PGA::Rendering::GlobalVars::getNumInstancedTriangleMeshes()=" + std::to_string(Rendering::GlobalVars::getNumInstancedTriangleMeshes()) + "]").c_str());
#endif
			}
#endif
			auto& instancedTriangleMesh = Rendering::GlobalVars::getInstancedTriangleMesh(terminalIndex);
			auto maxNumInstances = instancedTriangleMesh.maxNumInstances;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			auto i = atomicAdd(&instancedTriangleMesh.numInstances, 1) % maxNumInstances;
#else
			auto i = instancedTriangleMesh.numInstances % maxNumInstances;
			instancedTriangleMesh.numInstances++;
#endif
#ifdef PGA_COLUMN_MAJOR
			auto model = shape.getAdjustedModel() * math::float4x4::scale(shape.getSize());
#else
			auto model = transpose(shape.getAdjustedModel() * math::float4x4::scale(shape.getSize()));
#endif
			auto& instanceAttribute = instancedTriangleMesh.instancesAttributes[i];
			instanceAttribute.modelMatrix = model;
			instanceAttribute.custom = attr1;
		}

		template <typename ShapeT, typename... ArgsT>
		__host__ __device__ __inline__ static unsigned int allocateVertices(ShapeT& shape, unsigned int terminalIndex, ArgsT... args)
		{
			return 0;
		}

		template <typename... ArgsT>
		__host__ __device__ __inline__ static void perVertex(unsigned int vertexIndex, const math::float4& vertex, const math::float3& normal, const math::float2& uv, unsigned int terminalIndex, ArgsT... args)
		{
		}

		template <typename ShapeT, typename... ArgsT>
		__host__ __device__ __inline__ static unsigned int allocateIndices(ShapeT& shape, unsigned int terminalIndex, ArgsT... args)
		{
			return 0;
		}

		template <typename... ArgsT>
		__host__ __device__ __inline__ static void perTriangle(unsigned int index, unsigned int vertex0Index, unsigned int vertex1Index, unsigned int vertex2Index, unsigned int terminalIndex, ArgsT... args)
		{
		}

	};
	
	template <unsigned int IndexT>
	struct GenerationFunction < Rendering::GenFuncFilter, IndexT >
	{
		static_assert(IndexT <= 1, "invalid generation function index for PGA::Rendering::GenFuncFilter");
	
	};


}
