#pragma once

#if defined(PGA_D3D)
#include "D3DShapeMesh.h"
namespace PGA
{
	namespace Rendering
	{
		template <typename ShapeT>
		using ShapeMesh = typename PGA::Rendering::D3D::ShapeMeshImpl<ShapeT>;

	}

}
#else
#include "GLShapeMesh.h"
namespace PGA
{
	namespace Rendering
	{
		template <typename ShapeT>
		using ShapeMesh = typename PGA::Rendering::GL::ShapeMeshImpl<ShapeT>;

	}

}
#endif