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

<code>

	const unsigned int NumPhases = 1;

	struct Controller : Scenes::GriddedScene < <gridY>, <gridX> >
	{
		template <unsigned int AxiomId>
		struct AxiomTraits : Grid::DefaultAxiomTraits < Box, 0 >
		{
			__device__ __inline__ static math::float3 getSize(const Box& shape)
			{
				return math::float3(1.0f, 1.0f, 1.0f);
			}

		};

		typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

		struct TerminalsTraits : Rendering::SingleTerminalTraits < <maxNumVertices>, <maxNumIndices>, <maxNumInstances>, Box >
		{
		};

		static std::string name()
		{
			//return std::string("partition_<idx>_<uid>_") + std::to_string(getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
			return std::string("partition_<idx>_") + std::to_string(getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
		}
		
	};

}
