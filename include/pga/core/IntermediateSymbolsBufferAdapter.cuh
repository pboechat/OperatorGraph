#pragma once

#include "AABB.cuh"
#include "BVH.cuh"
#include "BoundingVolumeConstructor.cuh"
#include "CUDAException.h"
#include "Collision.cuh"
#include "ContextSensitivityDeviceVariables.cuh"
#include "IntermediateSymbol.cuh"
#include "IntermediateSymbolsBuffer.cuh"
#include "Shapes.cuh"

#include <cuda_runtime_api.h>

namespace PGA
{
	namespace ContextSensitivity
	{
		struct IntermediateSymbolsBufferAdapter
		{
		private:
			__host__ __device__ __inline__ static BVHMask fromRuleTagIdToMask(unsigned int ruleTagId)
			{
				return (((BVHMask)1) << ruleTagId);
			}

		public:
			__host__ static void initialize();
			__host__ static void reset();
			__host__ static void release();
			
			__host__ __device__ __inline__ static unsigned int getNumElements()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return ContextSensitivity::Device::IntermediateBoxes.counter + 
					ContextSensitivity::Device::IntermediateQuads.counter + 
					ContextSensitivity::Device::IntermediateSpheres.counter;
#else
				return ContextSensitivity::Host::IntermediateBoxes.counter +
					ContextSensitivity::Host::IntermediateQuads.counter +
					ContextSensitivity::Host::IntermediateSpheres.counter;
#endif
			}

			template <typename BoundingVolumeT>
			__host__ __device__ __inline__ static BVH<BoundingVolumeT> getLeafBVHNode(unsigned int i)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto l1 = ContextSensitivity::Device::IntermediateBoxes.counter;
				auto l2 = (ContextSensitivity::Device::IntermediateBoxes.counter + 
					ContextSensitivity::Device::IntermediateQuads.counter);
#else
				auto l1 = ContextSensitivity::Host::IntermediateBoxes.counter;
				auto l2 = (ContextSensitivity::Host::IntermediateBoxes.counter + 
					ContextSensitivity::Host::IntermediateQuads.counter);
#endif

				if (i < l1)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					auto& box = ContextSensitivity::Device::IntermediateBoxes.shapes[i];
#else
					auto& box = ContextSensitivity::Host::IntermediateBoxes.shapes[i];
#endif
					return BVH<BoundingVolumeT>(fromRuleTagIdToMask(box.colliderTag), BoundingVolumeConstructor<BoundingVolumeT>::construct(box));
				}
				else if (i < l2)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					auto& quad = ContextSensitivity::Device::IntermediateQuads.shapes[i - l1];
#else
					auto& quad = ContextSensitivity::Host::IntermediateQuads.shapes[i - l1];
#endif
					return BVH<BoundingVolumeT>(fromRuleTagIdToMask(quad.colliderTag), BoundingVolumeConstructor<BoundingVolumeT>::construct(quad));
				}
				else
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::ContextSensitivity::IntermediateSymbolsBufferAdapter::getLeafBVHNode(..): invalid BVH leaf node index [i=%d]", i);
					asm("trap;");
#else
					throw std::runtime_error("PGA::ContextSensitivity::IntermediateSymbolsBufferAdapter::getLeafBVHNode(..): invalid intermediate symbol index [i=" + std::to_string(i) + "]");
#endif
					return {};
				}
			}

			template <typename Shape>
			__host__ __device__ __inline__ static bool checkCollision(unsigned int i, unsigned int ruleTagId, const Shape& shape)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto l1 = ContextSensitivity::Device::IntermediateBoxes.counter;
				auto l2 = (ContextSensitivity::Device::IntermediateBoxes.counter +
					ContextSensitivity::Device::IntermediateQuads.counter);
#else
				auto l1 = ContextSensitivity::Host::IntermediateBoxes.counter;
				auto l2 = (ContextSensitivity::Host::IntermediateBoxes.counter +
					ContextSensitivity::Host::IntermediateQuads.counter);
#endif

				if (i < l1)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					auto& box = ContextSensitivity::Device::IntermediateBoxes.shapes[i];
#else
					auto& box = ContextSensitivity::Host::IntermediateBoxes.shapes[i];
#endif
					return box.colliderTag == ruleTagId && Collision::check(box, shape);
				}
				else if (i < l2)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					auto& quad = ContextSensitivity::Device::IntermediateQuads.shapes[i - l1];
#else
					auto& quad = ContextSensitivity::Host::IntermediateQuads.shapes[i - l1];
#endif
					return quad.colliderTag == ruleTagId && Collision::check(quad, shape);
				}
				else
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::ContextSensitivity::IntermediateSymbolsBufferAdapter::checkCollision(..): invalid intermediate symbol index [i=%d]", i);
					asm("trap;");
#else
					throw std::runtime_error("PGA::ContextSensitivity::IntermediateSymbolsBufferAdapter::checkCollision(..): invalid intermediate symbol index [i=" + std::to_string(i) + "]");
#endif
					return false;
				}
			}

		};

		template <typename Shape>
		struct PerShapeIntermediateSymbolsBufferAdapter
		{
			// NOTE: this empty declaration is necessary because in operator-based scheduling the compiler doesn't know
			// which shapes are actually going to be stored (it depends on the dynamic data stored in DispatchTable::ruleTagId), 
			// so the compiler ends up instantiating a ForShape type for each shape mentioned on the procedure list (see dispatch[...]() methods at CPU/GPUEvaluator.cuh)
			__host__ __device__ __inline__ static void store(int ruleTagId, const Shape& shape) { }

		};

		template <>
		struct PerShapeIntermediateSymbolsBufferAdapter <Shapes::Box>
		{
		private:
			static IntermediateSymbol<Shapes::Box>* shapes;

		public:
				__host__ static void initialize()
				{ 
					auto size = sizeof(IntermediateSymbolsBuffer<Shapes::Box>);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					PGA_CUDA_checkedCall(cudaMalloc((void**)&shapes, Constants::GetMaxIntermediateSymbols<Shapes::Box>::Result * size));
#else
					shapes = (IntermediateSymbol<Shapes::Box>*)malloc(Constants::GetMaxIntermediateSymbols<Shapes::Box>::Result * size);
#endif
					IntermediateSymbolsBuffer<Shapes::Box> intermediateBuffer;
					intermediateBuffer.counter = 0; 
					intermediateBuffer.shapes = shapes;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					PGA_CUDA_checkedCall(cudaMemcpyToSymbol((void*)&ContextSensitivity::Device::IntermediateBoxes, &intermediateBuffer, size));
#else
					memcpy((void*)&ContextSensitivity::Device::IntermediateBoxes, &intermediateBuffer, size);
#endif
				}

				__host__ static void release()
				{ 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					PGA_CUDA_checkedCall(cudaFree(shapes)); 
#else
					free(shapes);
#endif
				}

				__host__ static void reset()
				{ 
					unsigned int counter = 0; 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					PGA_CUDA_checkedCall(cudaMemcpyToSymbol((void*)&ContextSensitivity::Device::IntermediateBoxes, &counter, sizeof(unsigned int)));
#else
					memcpy((void*)&ContextSensitivity::Host::IntermediateBoxes, &counter, sizeof(unsigned int));
#endif
				}

				__host__ static unsigned int syncWithDeviceAndGetCounter()
				{ 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					IntermediateSymbolsBuffer<Shapes::Box> intermediateBuffer;
					PGA_CUDA_checkedCall(cudaMemcpyFromSymbol(&intermediateBuffer, (void*)&ContextSensitivity::Device::IntermediateBoxes, sizeof(IntermediateSymbolsBuffer<Shapes::Box>)));
					return intermediateBuffer.counter;
#else
					return ContextSensitivity::Host::IntermediateBoxes.counter;
#endif
				}

				__host__ __device__ __inline__ static unsigned int getCounter() 
				{ 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					return ContextSensitivity::Device::IntermediateBoxes.counter;
#else
					return ContextSensitivity::Host::IntermediateBoxes.counter;
#endif
				}

				__host__ __device__ __inline__ static void store(int colliderTag, const Shapes::Box& shape)
				{ 
					IntermediateSymbol<Shapes::Box> intermediateSymbol(shape);
					intermediateSymbol.colliderTag = colliderTag; 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					return ContextSensitivity::Device::IntermediateBoxes.store(intermediateSymbol);
#else
					return ContextSensitivity::Host::IntermediateBoxes.store(intermediateSymbol);
#endif
				}

				__host__ __device__ __inline__ static IntermediateSymbol<Shapes::Box>& getItem(unsigned int i) 
				{ 
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					return ContextSensitivity::Device::IntermediateBoxes.shapes[i]; 
#else
					return ContextSensitivity::Host::IntermediateBoxes.shapes[i];
#endif
				}

				__host__ __device__ __inline__ static unsigned int getOffset() 
				{ 
					return 0; 
				}

		};

		template <>
		struct PerShapeIntermediateSymbolsBufferAdapter <Shapes::Quad>
		{
		private:
			static IntermediateSymbol<Shapes::Quad>* shapes;

		public:
			__host__ static void initialize()
			{
				auto size = sizeof(IntermediateSymbolsBuffer<Shapes::Quad>);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				PGA_CUDA_checkedCall(cudaMalloc((void**)&shapes, Constants::GetMaxIntermediateSymbols<Shapes::Quad>::Result * size));
#else
				shapes = (IntermediateSymbol<Shapes::Quad>*)malloc(Constants::GetMaxIntermediateSymbols<Shapes::Quad>::Result * size);
#endif
				IntermediateSymbolsBuffer<Shapes::Quad> intermediateBuffer;
				intermediateBuffer.counter = 0;
				intermediateBuffer.shapes = shapes;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				PGA_CUDA_checkedCall(cudaMemcpyToSymbol((void*)&ContextSensitivity::Device::IntermediateQuads, &intermediateBuffer, size));
#else
				memcpy((void*)&ContextSensitivity::Device::IntermediateQuads, &intermediateBuffer, size);
#endif
			}

			__host__ static void release()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				PGA_CUDA_checkedCall(cudaFree(shapes));
#else
				free(shapes);
#endif
			}

			__host__ static void reset()
			{
				unsigned int counter = 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				PGA_CUDA_checkedCall(cudaMemcpyToSymbol((void*)&ContextSensitivity::Device::IntermediateQuads, &counter, sizeof(unsigned int)));
#else
				memcpy((void*)&ContextSensitivity::Host::IntermediateQuads, &counter, sizeof(unsigned int));
#endif
			}

			__host__ static unsigned int syncWithDeviceAndGetCounter()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				IntermediateSymbolsBuffer<Shapes::Quad> intermediateBuffer;
				PGA_CUDA_checkedCall(cudaMemcpyFromSymbol(&intermediateBuffer, (void*)&ContextSensitivity::Device::IntermediateQuads, sizeof(IntermediateSymbolsBuffer<Shapes::Quad>)));
				return intermediateBuffer.counter;
#else
				return ContextSensitivity::Host::IntermediateQuads.counter;
#endif
			}

			__host__ __device__ __inline__ static unsigned int getCounter()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return ContextSensitivity::Device::IntermediateQuads.counter;
#else
				return ContextSensitivity::Host::IntermediateQuads.counter;
#endif
			}

			__host__ __device__ __inline__ static void store(int colliderTag, const Shapes::Quad& shape)
			{
				IntermediateSymbol<Shapes::Quad> intermediateSymbol(shape);
				intermediateSymbol.colliderTag = colliderTag;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return ContextSensitivity::Device::IntermediateQuads.store(intermediateSymbol);
#else
				return ContextSensitivity::Host::IntermediateQuads.store(intermediateSymbol);
#endif
			}

			__host__ __device__ __inline__ static IntermediateSymbol<Shapes::Quad>& getItem(unsigned int i)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return ContextSensitivity::Device::IntermediateQuads.shapes[i];
#else
				return ContextSensitivity::Host::IntermediateQuads.shapes[i];
#endif
			}

			__host__ __device__ __inline__ static unsigned int getOffset()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return ContextSensitivity::Device::IntermediateBoxes.counter;
#else
				return ContextSensitivity::Host::IntermediateBoxes.counter;
#endif
			}

		};
		
		__host__ void IntermediateSymbolsBufferAdapter::initialize()
		{
			PerShapeIntermediateSymbolsBufferAdapter<Shapes::Box>::initialize();
			PerShapeIntermediateSymbolsBufferAdapter<Shapes::Quad>::initialize();
		}

		__host__ void IntermediateSymbolsBufferAdapter::reset()
		{
			PerShapeIntermediateSymbolsBufferAdapter<Shapes::Box>::reset();
			PerShapeIntermediateSymbolsBufferAdapter<Shapes::Quad>::reset();
		}

		__host__ void IntermediateSymbolsBufferAdapter::release()
		{
			PerShapeIntermediateSymbolsBufferAdapter<Shapes::Box>::release();
			PerShapeIntermediateSymbolsBufferAdapter<Shapes::Quad>::release();
		}

		IntermediateSymbol<Shapes::Box>* PerShapeIntermediateSymbolsBufferAdapter<Shapes::Box>::shapes = 0;
		IntermediateSymbol<Shapes::Quad>* PerShapeIntermediateSymbolsBufferAdapter<Shapes::Quad>::shapes = 0;

	}

}
