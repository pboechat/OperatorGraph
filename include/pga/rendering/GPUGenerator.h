#pragma once

#if defined(PGA_D3D)
#include "GPUD3DGenerator.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GPU::D3D::Generator;

	}

}
#else
#include "GPUGLGenerator.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GPU::GL::Generator;

	}

}
#endif