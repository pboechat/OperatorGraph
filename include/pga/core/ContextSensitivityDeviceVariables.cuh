#pragma once

#include "AABB.cuh"
#include "BVH.cuh"
#include "IntermediateSymbolsBuffer.cuh"
#include "RadixTree.h"
#include "Shapes.cuh"

#include <cuda_runtime_api.h>

namespace PGA
{
	namespace ContextSensitivity
	{
		namespace Device
		{
			__device__ ContextSensitivity::IntermediateSymbolsBuffer<Shapes::Box> IntermediateBoxes;
			__device__ ContextSensitivity::IntermediateSymbolsBuffer<Shapes::Quad> IntermediateQuads;
			__device__ ContextSensitivity::IntermediateSymbolsBuffer<Shapes::Sphere> IntermediateSpheres;
			__device__ ContextSensitivity::BVH<ContextSensitivity::AABB>* BVHNodes;
			__device__ ContextSensitivity::RadixTree* BVHStructure;
			__device__ unsigned int NumCollisionChecks;
		}

		namespace Host
		{
			ContextSensitivity::IntermediateSymbolsBuffer<Shapes::Box> IntermediateBoxes;
			ContextSensitivity::IntermediateSymbolsBuffer<Shapes::Quad> IntermediateQuads;
			ContextSensitivity::IntermediateSymbolsBuffer<Shapes::Sphere> IntermediateSpheres;
			ContextSensitivity::BVH<ContextSensitivity::AABB>* BVHNodes;
			ContextSensitivity::RadixTree* BVHStructure;
			unsigned int NumCollisionChecks;
		}


	}

}
