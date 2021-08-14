#pragma once

#include "AABB.cuh"
#include "BVH.cuh"
#include "CUDAException.h"
#include "GlobalVariables.cuh"
#include "IntermediateSymbolsBufferAdapter.cuh"
#include "RadixTree.h"

#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <math/math.h>
#include <math/vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cstdint>
#include <stdexcept>

#define DIVIDE_AND_ROUND_UP(x, y) ((x) + (y) - 1) / (y)
#define COORD_LIM ((1 << 10)-1)

namespace PGA
{
	namespace ContextSensitivity
	{
		namespace GPU
		{
			typedef unsigned long long MortonCode;
			const unsigned int GenerateMortonCodesAndBoudingVolumesKernelBlockSize = 256;
			const unsigned int BuildRadixTreeKernelBlockSize = 256;
			const unsigned int PropagateBVHKernelBlockSize = 256;

			__inline__ __device__ int delta(MortonCode& A, MortonCode& B)
			{
				return __clzll(A ^ B);
			}

			__inline__ __device__ int sign(int v)
			{
				return -1 + 2 * (v >= 0);
			}

			__device__ __inline__ void spread(unsigned int& v)
			{
				v = (v | (v << 16)) & 0x030000FF;
				v = (v | (v << 8)) & 0x0300F00F;
				v = (v | (v << 4)) & 0x030C30C3;
				v = (v | (v << 2)) & 0x09249249;
			}

			__device__ void morton30(math::float3& normalizedCenter, unsigned int& code)
			{
				unsigned int v;
				v = (normalizedCenter.x * COORD_LIM);
				spread(v);
				code = v;
				v = (normalizedCenter.y * COORD_LIM);
				spread(v);
				code |= (v << 1);
				v = (normalizedCenter.z * COORD_LIM);
				spread(v);
				code |= (v << 2);
			}

			template <typename Shape>
			__global__ void generateMortonCodes(math::float3 boundaryMin, math::float3 boundarySize, MortonCode* mortonCodes)
			{
				unsigned int i = (blockIdx.x * GenerateMortonCodesAndBoudingVolumesKernelBlockSize) + threadIdx.x;

				if (i >= PerShapeIntermediateSymbolsBufferAdapter<Shape>::getCounter())
				{
					return;
				}

				math::float3 center = PerShapeIntermediateSymbolsBufferAdapter<Shape>::getItem(i).getPosition();
				center = (center - boundaryMin);
				center = math::float3(center.x / boundarySize.x, center.y / boundarySize.y, center.z / boundarySize.z);

				unsigned int mortonCode;
				morton30(center, mortonCode);

				unsigned int j = PerShapeIntermediateSymbolsBufferAdapter<Shape>::getOffset() + i;

				mortonCodes[j] = (((MortonCode)mortonCode) << 32) | j;
			}

			__global__ void buildRadixTree(MortonCode* codes, unsigned int numElements, RadixTree* result, unsigned int* leafParents)
			{
				int local = threadIdx.x;
				int off = (blockIdx.x * BuildRadixTreeKernelBlockSize);
				int i = off + local;
				int j;

				int t1, t2;
				int d;

				int delta_min;

				MortonCode ni;
				MortonCode n;

				// load to shared memory

				__shared__ MortonCode sMortonCodes[BuildRadixTreeKernelBlockSize + 2];

				t1 = min(BuildRadixTreeKernelBlockSize, numElements - off - 1);

				if (i && local < 2)
				{
					sMortonCodes[local*(t1 + 1)] = codes[off - 1 + local*(t1 + 1)];
				}

				if (local >= t1)
				{
					return;
				}

				ni = codes[i];
				sMortonCodes[local + 1] = ni;

				__syncthreads();

				// determine direction

				if (!i)
				{
					d = 1;
					t2 = (numElements - 1);
				}
				else
				{
					n = sMortonCodes[local];
					t1 = delta(ni, n);

					n = sMortonCodes[local + 2];
					t2 = delta(ni, n);

					delta_min = min(t1, t2);
					d = sign(t2 - t1);

					// bound end

					t1 = 128;	// apparently better performance
					while ((j = i + (d*t1)) >= 0 && j < numElements && delta(ni, codes[j]) > delta_min)
					{
						t1 <<= 2;
					}

					// determine end

					t2 = 0;
					for (t1 >>= 1; t1 > 0; t1 >>= 1)
					{
						j = i + (t2 + t1)*d;
						if (j < 0 || j >= numElements)
						{
							continue;
						}

						n = codes[j];
						if (delta(ni, n) > delta_min)
						{
							t2 += t1;
						}
					}
				}

				j = i + t2*d;
				n = codes[j];
				delta_min = delta(ni, n);

				// determine split

				t1 = (1 << max(0, 31 - __clz(t2)));
				t1 >>= (t1 == t2);						 // go to next lower power of 2

				t2 = 0;
				for (; t1 > 0; t1 >>= 1)
				{
					n = codes[i + (t2 + t1)*d];
					if (delta(ni, n) > delta_min)
					{
						t2 += t1;
					}
				}

				d = i + t2*d + min(d, 0);

				t1 = ((min(i, j) == d) ? d : -d);
				t2 = ((max(i, j) == (d + 1)) ? (d + 1) : -(d + 1));

				result[i].left = t1;
				result[i].right = t2;

				if (t1 < 0)
				{
					result[-t1].parent = i;
				}
				else
				{
					leafParents[t1] = i;
				}

				if (t2 < 0)
				{
					result[-t2].parent = i;
				}
				else
				{
					leafParents[t2] = i;
				}
			}

			template <typename BoundingVolume>
			__global__ void propagateBVH(unsigned int numElements, RadixTree* BVHStructure, unsigned int* BVHLeafParents, BVH<BoundingVolume>* BVHNodes, unsigned int* threadCounts)
			{
				int current = (blockIdx.x * PropagateBVHKernelBlockSize) + threadIdx.x;

				if (current >= numElements)
				{
					return;
				}

				BVH<BoundingVolume> element;
				RadixTree entry;
				current = BVHLeafParents[current];
				while (atomicInc(threadCounts + current, INT_MAX))
				{
					entry = BVHStructure[current];

					if (entry.left < 0)
					{
						element = BVHNodes[-entry.left];
					}
					else
					{
						element = IntermediateSymbolsBufferAdapter::template getLeafBVHNode<BoundingVolume>(entry.left);
					}

					if (entry.right < 0)
					{
						element.propagate(BVHNodes[-entry.right]);
					}
					else
					{
						element.propagate(IntermediateSymbolsBufferAdapter::template getLeafBVHNode<BoundingVolume>(entry.right));
					}

					BVHNodes[current] = element;

					if (current)
					{
						current = entry.parent;
					}
					else
					{
						break;
					}
				}
			}

			struct BVHConstructor
			{
				static double construct(const math::float3& boundaryMin, const math::float3& boundaryMax)
				{
					unsigned int numBoxes = PerShapeIntermediateSymbolsBufferAdapter<Shapes::Box>::syncWithDeviceAndGetCounter();
					unsigned int numQuads = PerShapeIntermediateSymbolsBufferAdapter<Shapes::Quad>::syncWithDeviceAndGetCounter();
					unsigned int numElements = numBoxes + numQuads;

					double executionTime = 0.0;
					RadixTree* dBVHStructure = 0;
					BVH<AABB>* dBVHNodes = 0;

					if (numElements < 2)
					{
						PGA_CUDA_checkedCall(cudaMemcpyToSymbol(ContextSensitivity::Device::BVHStructure, &dBVHStructure, sizeof(RadixTree*)));
						PGA_CUDA_checkedCall(cudaMemcpyToSymbol(ContextSensitivity::Device::BVHNodes, &dBVHNodes, sizeof(AABB*)));
					}
					else
					{
						unsigned int bvhDepth = (unsigned int)ceil(log((double)numElements) / log(2.0)) + 1;

						// FIXME: checking boundaries
						if (bvhDepth > ContextSensitivity::Constants::MaxBVHDepth)
						{
							throw std::runtime_error("bvhDepth > ContextSensitivity::Constants::MaxBVHDepth");
						}

						math::float3 boundarySize = boundaryMax - boundaryMin;

						MortonCode* dMortonCodes;
						unsigned int* dBVHLeafParents;
						unsigned int* dThreadCounts;
						PGA_CUDA_checkedCall(cudaMalloc((void**)&dMortonCodes, sizeof(MortonCode) * numElements));
						PGA_CUDA_checkedCall(cudaMalloc((void**)&dBVHStructure, sizeof(RadixTree) * (numElements - 1)));
						PGA_CUDA_checkedCall(cudaMalloc((void**)&dBVHLeafParents, sizeof(unsigned int) * numElements));
						PGA_CUDA_checkedCall(cudaMalloc((void**)&dBVHNodes, sizeof(BVH<AABB>) * (numElements - 1)));
						PGA_CUDA_checkedCall(cudaMalloc((void**)&dThreadCounts, sizeof(unsigned int) * (numElements - 1)));
						PGA_CUDA_checkedCall(cudaMemset(dThreadCounts, 0, sizeof(unsigned int) * (numElements - 1)));

						cudaEvent_t startEvent, stopEvent;
						PGA_CUDA_checkedCall(cudaEventCreate(&startEvent));
						PGA_CUDA_checkedCall(cudaEventCreate(&stopEvent));

						PGA_CUDA_checkedCall(cudaEventRecord(startEvent, 0));

						generateMortonCodes<Shapes::Box> <<<DIVIDE_AND_ROUND_UP(numBoxes, GenerateMortonCodesAndBoudingVolumesKernelBlockSize), GenerateMortonCodesAndBoudingVolumesKernelBlockSize>>>(boundaryMin, boundarySize, dMortonCodes);
						generateMortonCodes<Shapes::Quad> <<<DIVIDE_AND_ROUND_UP(numQuads, GenerateMortonCodesAndBoudingVolumesKernelBlockSize), GenerateMortonCodesAndBoudingVolumesKernelBlockSize>>>(boundaryMin, boundarySize, dMortonCodes);

						thrust::device_ptr<MortonCode> mortonCodesPtr = thrust::device_ptr<MortonCode>(dMortonCodes);
						thrust::sort(mortonCodesPtr, mortonCodesPtr + numElements);

						buildRadixTree << <DIVIDE_AND_ROUND_UP(numElements, BuildRadixTreeKernelBlockSize), BuildRadixTreeKernelBlockSize >> >(dMortonCodes, numElements, dBVHStructure, dBVHLeafParents);

						propagateBVH <<<DIVIDE_AND_ROUND_UP(numElements, PropagateBVHKernelBlockSize), PropagateBVHKernelBlockSize>>>(numElements, dBVHStructure, dBVHLeafParents, dBVHNodes, dThreadCounts);

						PGA_CUDA_checkedCall(cudaEventRecord(stopEvent, 0));
						PGA_CUDA_checkedCall(cudaEventSynchronize(stopEvent));
						float elapsedTime;
						PGA_CUDA_checkedCall(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
						executionTime = elapsedTime / 1000.0;

						PGA_CUDA_checkedCall(cudaFree(dThreadCounts));

						PGA_CUDA_checkedCall(cudaMemcpyToSymbol(ContextSensitivity::Device::BVHStructure, &dBVHStructure, sizeof(RadixTree*)));
						PGA_CUDA_checkedCall(cudaMemcpyToSymbol(ContextSensitivity::Device::BVHNodes, &dBVHNodes, sizeof(AABB*)));
					}

					return executionTime;
				}

			};

		}

	}

}
