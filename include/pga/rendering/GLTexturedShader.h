#pragma once

#include "GLMaterial.h"
#include "GLShader.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class TexturedShader : public PGA::Rendering::GL::Shader
			{
			protected:
				virtual void setDefaultParameters(PGA::Rendering::GL::Material& material)
				{
					material.setParameter("useUv0", 1.0f);
					material.setParameter("uvScale0", math::float2(1.0f, 1.0f));
					material.setParameter("color0", math::float4(1.0f, 1.0f, 1.0f, 1.0f));
				}

			public:
				TexturedShader() : PGA::Rendering::GL::Shader("PGA_Rendering_TexturedShader.vs.glsl", "PGA_Rendering_TexturedShader.fs.glsl") {}
				TexturedShader(const TexturedShader&) = delete;
				TexturedShader& operator=(TexturedShader&) = delete;

			};

		}

	}

}