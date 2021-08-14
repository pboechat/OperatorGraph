#pragma once

#include <cuda_runtime_api.h>
#include <pga/core/Core.h>
#include <pga/core/Grid.cuh>

#include <map>
#include <string>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;

namespace Scene
{
	const std::map<std::size_t, std::size_t> GenFuncCounters = <genFuncCounters>;
	const unsigned int NumEdges = <numEdges>;
	const unsigned int NumSubgraphs = <numSubgraphs>;
	const bool Instrumented = <instrumented>;

<code>

	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, 0 >
	{
	};

	typedef Grid::Static<AxiomTraits, 1, 1, 1> AxiomGenerator;

	struct Controller
	{
		static const bool IsInitializable = false;
		static const bool IsGridded = false;
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

	std::string name()
	{
		return std::string("partition_<idx>_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
	}

}