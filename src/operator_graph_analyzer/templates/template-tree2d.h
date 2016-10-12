#pragma once

#include "ScenesCommons.cuh"

#include <pga/core/GlobalConstants.h>
#include <pga/core/DispatchTable.h>
#include <pga/core/Proc.cuh>
#include <pga/core/Operators.cuh>
#include <pga/core/Shapes.cuh>
#include <pga/core/Parameters.cuh>
#include <pga/core/Axis.h>
#include <pga/core/TStdLib.h>
#include <pga/core/Grid.cuh>
#include <pga/rendering/SingleTerminalTraits.cuh>

#include <math/math.h>

#include <cuda_runtime_api.h>

#include <string>
#include <map>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;
using namespace PGA::AxiomGenerators;

namespace Scene
{
	const unsigned int QueueSize = <avgQueueSize>;
	const std::map<std::size_t, std::size_t> GenFuncCounters = <genFuncCounters>;
	const unsigned int NumEdges = <numEdges>;
	const unsigned int NumSubgraphs = <numSubgraphs>;
	const bool Instrumented = <instrumented>;
	static const unsigned int MaxIterations = 9;
	static const unsigned int NumBranches = T::Power<2, MaxIterations>::Result + 1;
	static const unsigned int MaxNumAxioms = NumBranches * <gridY> * <gridX>;
	static const unsigned int MaxNumVertices = MaxNumAxioms * 24;
	static const unsigned int MaxNumIndices = MaxNumAxioms * 36;
	static const unsigned int MaxNumInstances = MaxNumAxioms;

<code>

	const unsigned int NumPhases = 1;

	struct Controller : Scenes::GriddedScene < <gridY>, <gridX> >, Scenes::SceneWithAttributes< 1 >, Scenes::IInitializableScene
	{
		template <unsigned int AxiomId>
		struct AxiomTraits : Grid::DefaultAxiomTraits < Box, 0 >
		{
			__device__ __inline__ static math::float3 getSize(const Box& shape)
			{
				float height = math::max(1.0f, math::pow(2.0f, (getAttribute(0) % MaxIterations))) * 0.1f;
				return math::float3(0.025f, height, 0.025f);
			}
			
			__device__ __inline__ static math::float3 getPosition(int row, int column, const Shape& shape)
			{
				float offset = shape.getSize().y * 0.5f;
				return math::float3(column * offset, offset, row * offset);
			}

		};

		typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

		struct TerminalsTraits : Rendering::SingleTerminalTraits < MaxNumVertices, MaxNumIndices, MaxNumInstances, Box >
		{
		};

		static void initialize()
		{
			setAttribute(0, 8);
		}
		
		static std::string name()
		{
			//return std::string("partition_<idx>_<uid>_") + std::to_string(getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
			return std::string("partition_<idx>_") + std::to_string(getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
		}
		
	};

}
