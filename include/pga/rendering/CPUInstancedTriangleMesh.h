#pragma once

#if defined(PGA_D3D)
#include "CPUD3DInstancedTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::CPU::D3D::InstancedTriangleMesh;

	}

}
#else
#include "CPUGLInstancedTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::CPU::GL::InstancedTriangleMesh;

	}

}
#endif