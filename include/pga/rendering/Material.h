#pragma once

#if defined(PGA_D3D)
#include "D3DMaterial.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::Material;

	}

}
#else
#include "GLMaterial.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::Material;

	}

}
#endif