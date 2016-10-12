#pragma once

#if defined(PGA_D3D)
#error "PGA::Rendering::D3D::DDS is not implemented yet"
#else
#include "GLDDS.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::DDS;

	}

}
#endif