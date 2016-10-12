#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "ContextSensitivityDeviceVariables.cuh"
#include "Shapes.cuh"
#include "RadixTree.h"
#include "Collision.cuh"
#include "IntermediateSymbolsBufferAdapter.cuh"
#include "TStdLib.h"

namespace PGA
{
	namespace ContextSensitivity
	{
		struct BVHTraversal
		{
			template <typename Shape>
			__host__ __device__ static bool checkCollision(unsigned char ruleTagId, const Shape& shape)
			{
				unsigned int numElements = IntermediateSymbolsBufferAdapter::getNumElements();
				// FIXME: workaround weird CUDA behaviour
				bool myBugFixBool = false;
				if (numElements == 0)
				{
					if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("[BVHTraversal] No collision (Empty BVH)\n");
#else
						std::cout << "[BVHTraversal] No collision (Empty BVH)" << std::endl;
#endif
					return false;
				}
				else if (numElements == 1)
				{
					if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("[BVHTraversal] Checking collision with root (Single element BVH)\n");
#else
						std::cout << "[BVHTraversal] Checking collision with root (Single element BVH)" << std::endl;
#endif
					myBugFixBool = IntermediateSymbolsBufferAdapter::checkCollision(0, ruleTagId, shape);
					return myBugFixBool;
				}
				else
				{
					RadixTree stack[ContextSensitivity::Constants::MaxBVHDepth];
					int stackTail = -1;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					stack[++stackTail] = ContextSensitivity::Device::BVHStructure[0];
#else
					stack[++stackTail] = ContextSensitivity::Host::BVHStructure[0];
#endif
					while (stackTail >= 0)
					{
						RadixTree entry = stack[stackTail--];
						if (entry.left < 0)
						{
							if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
								printf("[BVHTraversal] Checking collision with inner node %d (left)\n", -entry.left);
#else
								std::cout << "[BVHTraversal] Checking collision with inner node " << -entry.left << " (left)\n" << std::endl;
#endif
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							if (Collision::check(ContextSensitivity::Device::BVHNodes[-entry.left].boundingVolume, shape))
								stack[++stackTail] = ContextSensitivity::Device::BVHStructure[-entry.left];
#else
							if (Collision::check(ContextSensitivity::Host::BVHNodes[-entry.left].boundingVolume, shape))
								stack[++stackTail] = ContextSensitivity::Host::BVHStructure[-entry.left];
#endif
						}
						else
						{
							if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
								printf("[BVHTraversal] Checking collision with leaf node %d (left)\n", entry.left);
#else
								std::cout << "[BVHTraversal] Checking collision with leaf node " << entry.left << " (left)" << std::endl;
#endif
							myBugFixBool = IntermediateSymbolsBufferAdapter::checkCollision(entry.left, ruleTagId, shape);
							if (myBugFixBool == true)
							{
								if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
									printf("[BVHTraversal] Checking collision with leaf node %d (left)\n", entry.left);
#else
									std::cout << "[BVHTraversal] Checking collision with leaf node " << entry.left << " (left)" << std::endl;
#endif
								return myBugFixBool;
							}
						}
						if (entry.right < 0)
						{
							if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
								printf("[BVHTraversal] Checking collision with inner node %d (right)\n", -entry.right);
#else
								std::cout << "[BVHTraversal] Checking collision with inner node " << -entry.right << " (right)" << std::endl;
#endif
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							if (Collision::check(ContextSensitivity::Device::BVHNodes[-entry.right].boundingVolume, shape))
								stack[++stackTail] = ContextSensitivity::Device::BVHStructure[-entry.right];
#else
							if (Collision::check(ContextSensitivity::Host::BVHNodes[-entry.right].boundingVolume, shape))
								stack[++stackTail] = ContextSensitivity::Host::BVHStructure[-entry.right];
#endif
						}
						else
						{
							if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
								printf("[BVHTraversal] Checking collision with leaf node %d (right)\n", entry.right);
#else
								std::cout << "[BVHTraversal] Checking collision with leaf node " << entry.right << " (right)" << std::endl;
#endif
							myBugFixBool = IntermediateSymbolsBufferAdapter::checkCollision(entry.right, ruleTagId, shape);
							if (myBugFixBool)
							{
								if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
									printf("[BVHTraversal] Checking collision with leaf node %d (right)\n", entry.right);
#else
									std::cout << "[BVHTraversal] Checking collision with leaf node " << entry.right << " (right)" << std::endl;
#endif
								return myBugFixBool;
							}
						}
					}
					if (T::IsEnabled<DebugFlags::BVHTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("[BVHTraversal] No Collision with Collider(%d) detected\n", ruleTagId);
#else
						std::cout << "[BVHTraversal] No Collision with Collider(" << ruleTagId << ") detected" << std::endl;
#endif
					return myBugFixBool;
				}
			}

		};

	}

}
