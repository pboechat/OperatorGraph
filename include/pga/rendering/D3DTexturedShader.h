#pragma once

#include "D3DMaterial.h"
#include "D3DShader.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class TexturedShader : public PGA::Rendering::D3D::Shader
			{
			protected:
				virtual void setDefaultParameters(PGA::Rendering::D3D::Material& material)
				{
					material.setParameter("alphaThreshold", 0.0f);
					material.setParameter("useColorLookup", 1.0f);
					material.setParameter("useUvX0", 1.0f);
					material.setParameter("useUvY0", 1.0f);
					material.setParameter("uvScale0", math::float2(1.0f, 1.0f));
					material.setParameter("color0", math::float4(1.0f, 1.0f, 1.0f, 1.0f));
				}

			public:
				TexturedShader(ID3D11Device* device) : PGA::Rendering::D3D::Shader(device, "PGA_Rendering_TexturedShader.vs.hlsl", "PGA_Rendering_TexturedShader.ps.hlsl") {}
				TexturedShader(const TexturedShader&) = delete;
				TexturedShader& operator=(TexturedShader&) = delete;

			};

		}

	}

}