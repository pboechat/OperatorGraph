#pragma once

#if defined(PGA_D3D)
#include "D3DColorShader.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::ColorShader;

	}
}
#else
#include "GLColorShader.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::ColorShader;

	}
}
#endif
