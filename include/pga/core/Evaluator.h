#pragma once

#if defined(PGA_CPU)
#include "CPUEvaluator.h"
namespace PGA
{
	using PGA::CPU::SinglePhaseEvaluator;
	using PGA::CPU::MultiPhaseEvaluator;

}
#else
#include "GPUEvaluator.cuh"
namespace PGA
{
	using PGA::GPU::SinglePhaseEvaluator;
	using PGA::GPU::MultiPhaseEvaluator;

}
#endif
