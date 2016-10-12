#pragma once

#include "TStdLib.h"

namespace PGA
{
	namespace Constants
	{
		const unsigned int MaxNumParameters = 17; // if smaller than (Device::Constants::MaxNumSides + 1), might run into problems with SetAsDynamic(...)
		const unsigned int MaxNumSides = 16;
		const unsigned int MaxNumExpLevels = 4;
		const unsigned int MaxNumParameterValues = T::Power<2, MaxNumExpLevels>::Result;
		const unsigned int MaxNumSuccessors = 20;
		const unsigned int MaxNumAxioms = 32;
		const unsigned int NumSphereSlices = 30;

	}

}
