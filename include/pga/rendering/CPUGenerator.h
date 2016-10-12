#pragma once

#if defined(PGA_D3D)
#include "CPUD3DGenerator.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::CPU::D3D::Generator;

	}

}
#else
#include "CPUGLGenerator.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::CPU::GL::Generator;

	}

}
#endif