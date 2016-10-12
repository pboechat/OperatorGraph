#pragma once

#if defined(PGA_D3D)
#include "CPUD3DTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::CPU::D3D::TriangleMesh;

	}

}
#else
#include "CPUGLTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::CPU::GL::TriangleMesh;

	}

}
#endif