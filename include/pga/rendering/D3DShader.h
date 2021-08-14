#pragma once

#include "CameraParameters.h"
#include "D3DMaterial.h"

#include <d3d11.h>
#include <math/matrix.h>
#include <math/vector.h>

#include <initializer_list>
#include <memory>
#include <string>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class Shader
			{
			private:
				Shader(const Shader&) = delete;
				Shader& operator =(Shader&) = delete;

			protected:
				ID3D11VertexShader* vertexShader;
				ID3D11PixelShader* pixelShader;
				ID3D11InputLayout* inputLayout;

				virtual void setDefaultParameters(PGA::Rendering::D3D::Material& material) {}
				ID3D11InputLayout* createInputLayout(ID3D11Device* device, ID3D10Blob* shaderBinary);

				friend PGA::Rendering::D3D::Material;

			public:
				static void setIncludePath(const std::initializer_list<std::string>& includePath);

				Shader() = delete;
				virtual ~Shader();
				Shader(ID3D11Device* device, ID3D10Blob* vertexShaderBinary, ID3D10Blob* pixelShaderBinary);
				Shader(ID3D11Device* device, const std::string& vertexShaderFileName, const std::string& pixelShaderFileName);
				Shader(Shader&& other);
				Shader& operator=(Shader&& other);

				void bind(ID3D11DeviceContext* deviceContext);

			};

		}

	}

}
