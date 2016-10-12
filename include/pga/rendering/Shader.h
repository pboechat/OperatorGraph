#pragma once

#if defined(PGA_D3D)
#include "D3DShader.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::Shader;

	}
}
#else
#include "GLShader.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::Shader;

	}
}
#endif
