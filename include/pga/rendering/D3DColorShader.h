#pragma once

#include "D3DMaterial.h"
#include "D3DShader.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class ColorShader : public PGA::Rendering::D3D::Shader
			{
			protected:
				virtual void setDefaultParameters(PGA::Rendering::D3D::Material& material)
				{
					material.setParameter("color0", math::float4(1.0f, 1.0f, 1.0f, 1.0f));
				}

			public:
				ColorShader(ID3D11Device* device) : PGA::Rendering::D3D::Shader(device, "PGA_Rendering_ColorShader.vs.hlsl", "PGA_Rendering_ColorShader.ps.hlsl") {}
				ColorShader(const ColorShader&) = delete;
				ColorShader& operator=(ColorShader&) = delete;

			};

		}

	}

}

