#pragma once

#include <map>
#include <string>
#include <cuda_runtime_api.h>

#include <pga/core/Core.h>
#include <pga/core/Grid.cuh>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;

namespace Scene
{
	const std::map<size_t, size_t> GenFuncCounters = <genFuncCounters>;
	const unsigned int NumEdges = <numEdges>;
	const unsigned int NumSubgraphs = <numSubgraphs>;
	const bool Instrumented = <instrumented>;
    const size_t Idx = <idx>;
    const std::string Uid = "<uid>";
    const unsigned int QueueSize = <queueSize>;
    const PGA::GPU::Technique Technique = <technique>;

<code>

	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, <operatorCode> >
	{
		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(16, 16, 16);
		}
		
		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3(column * 64.0f - 8.0f, 0.0f, row * 64.0f - 8.0f);
		}
		
		__host__ __device__ __inline__ static int getEntryIndex()
		{
			return <entryIdx>;
		}
            
	};

	typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

	struct Controller : SceneUtils::GriddedSceneController< <gridY>, <gridX> >
	{
		static const bool IsInitializable = false;
		static const bool HasAttributes = false;

	};

	struct Configurator
	{
		static const bool IsFile = true;
		static std::string value()
		{
			return "configuration.xml";
		}

	};

	std::string testName()
	{
		return std::string("partition_<idx>_") + std::to_string(Grid::GlobalVars::getNumElements()) + ("_o" + std::to_string(<optimization>)) + ((Instrumented) ? "_i" : "");
	}
	
	std::string sceneName()
	{
		return std::string("partition_<idx>") + ("_o" + std::to_string(<optimization>)) + ((Instrumented) ? "_i" : "");
	}
    
}