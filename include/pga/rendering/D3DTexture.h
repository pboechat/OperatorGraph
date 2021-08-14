#pragma once

#include "Image.h"

#include <d3d11.h>

#include <string>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class Texture
			{
			protected:
				ID3D11Resource* resource;
				ID3D11SamplerState* sampler;
				ID3D11ShaderResourceView* shaderResourceView;

			public:
				Texture(ID3D11Device* device, ID3D11DeviceContext* deviceContext, const std::string fileName);
				virtual ~Texture();
				Texture(const Texture&&);
				Texture(const Texture&) = delete;
				Texture& operator=(const Texture&) = delete;
				Texture& operator=(const Texture&&);

				void bind(ID3D11DeviceContext* deviceContext, unsigned int slot) const;

			};

		}

	}

}