#pragma once

#if defined(PGA_D3D)
#include "GPUD3DInstancedTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GPU::D3D::InstancedTriangleMesh;

	}

}
#else
#include "GPUGLInstancedTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GPU::GL::InstancedTriangleMesh;

	}

}
#endif