#pragma once

#if defined(PGA_D3D)
#include "D3DTexturedShader.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::TexturedShader;

	}
}
#else
#include "GLTexturedShader.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::GL::TexturedShader;

	}
}
#endif
