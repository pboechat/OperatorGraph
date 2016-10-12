#pragma once

#include "GlobalConstants.h"
#include "ParameterType.h"

namespace PGA
{
	struct DispatchTableEntry
	{
		struct Parameter
		{
			float value[PGA::Constants::MaxNumParameterValues];
			ParameterType type;

		};

		struct Successor
		{
			int entryIndex;
			int phaseIndex;

			Successor() : entryIndex(-1), phaseIndex(-1) {}

		};

		int operatorCode;
		unsigned int numParameters;
		Parameter parameters[PGA::Constants::MaxNumParameters];
		unsigned int numSuccessors;
		unsigned char ruleTagId;
		Successor successors[PGA::Constants::MaxNumSuccessors];
		unsigned int numEdgeIndices;
		int edgeIndices[PGA::Constants::MaxNumSuccessors];
		int subGraphIndex;

	};

}
