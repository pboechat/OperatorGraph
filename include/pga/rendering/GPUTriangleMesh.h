#pragma once

#if defined(PGA_D3D)
#include "GPUD3DTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GPU::D3D::TriangleMesh;

	}

}
#else
#include "GPUGLTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GPU::GL::TriangleMesh;

	}

}
#endif