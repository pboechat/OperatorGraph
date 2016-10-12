#pragma once

#if defined(PGA_D3D)
#include "D3DMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::Mesh;

	}

}
#else
#include "GLMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::Mesh;

	}

}
#endif