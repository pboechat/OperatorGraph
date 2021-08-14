#pragma once

#include <pga/compiler/Graph.h>

#include <ostream>

namespace PGA
{
	namespace Compiler
	{
		class CodeGenerator
		{
		public:
			CodeGenerator() = delete;

			static void fromPartition(std::ostream& out, const Graph::PartitionPtr& partition/*, bool optimized = false, bool instrumented = false*/);
			static void fromPartition(std::ostream& out, const Graph::PartitionPtr& partition, bool optimized, bool instrumented);
			static void fromPartition(std::ostream& out, const Graph::PartitionPtr& partition, bool& staticFirstProcedure);
			static void fromPartition(std::ostream& out, const Graph::PartitionPtr& partition, bool optimized, bool instrumented, bool& staticFirstProcedure);

		};

	}

}