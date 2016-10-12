#pragma once

#if defined(PGA_D3D)
#include "D3DInstancedTriangleMeshSource.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::InstancedTriangleMeshSource;

	}

}
#else
#include "GLInstancedTriangleMeshSource.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::InstancedTriangleMeshSource;

	}

}
#endif