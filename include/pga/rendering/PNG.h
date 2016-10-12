#pragma once

#if defined(PGA_D3D)
#error "PGA::Rendering::D3D::PNG is not implemented yet"
#else
#include "GLPNG.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::PNG;

	}

}
#endif