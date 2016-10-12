#pragma once

#if defined(PGA_D3D)
#include "D3DTexture.h"
namespace PGA
{
	namespace Rendering
	{
		using PGA::Rendering::D3D::Texture;

	}

}
#else
#include "GLTexture.h"
namespace PGA 
{
	namespace Rendering 
	{
		using PGA::Rendering::GL::Texture;
		using PGA::Rendering::GL::Texture1D;
		using PGA::Rendering::GL::Texture2D;

	}

}
#endif