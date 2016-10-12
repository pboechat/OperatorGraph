#pragma once

#if defined(PGA_D3D)
#include "D3DOBJMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::OBJMesh;

	}

}
#else
#include "GLOBJMesh.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::OBJMesh;

	}

}
#endif