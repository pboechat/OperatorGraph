#pragma once

#include "GLMaterial.h"
#include "GLShader.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class ColorShader : public PGA::Rendering::GL::Shader
			{
			protected:
				virtual void setDefaultParameters(PGA::Rendering::GL::Material& material)
				{
					material.setParameter("color0", math::float4(1.0f, 1.0f, 1.0f, 1.0f));
				}

			public:
				ColorShader() : PGA::Rendering::GL::Shader("PGA_Rendering_ColorShader.vs.glsl", "PGA_Rendering_ColorShader.fs.glsl") {}
				ColorShader(const ColorShader&) = delete;
				ColorShader& operator=(ColorShader&) = delete;

			};

		}

	}

}

