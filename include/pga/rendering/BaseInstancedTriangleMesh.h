#pragma once

#if defined(PGA_D3D)
#include "D3DBaseInstancedTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::BaseInstancedTriangleMesh;

	}
}
#else
#include "GLBaseInstancedTriangleMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::BaseInstancedTriangleMesh;

	}
}
#endif
