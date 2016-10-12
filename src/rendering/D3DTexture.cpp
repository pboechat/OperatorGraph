#include <string>
#include <memory>

#include "DDSTextureLoader.h"

#include <pga/rendering/D3DException.h>

#include "../include/pga/rendering/D3DTexture.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			Texture::Texture(ID3D11Device* device, ID3D11DeviceContext* deviceContext, const std::string fileName)
			{
				DirectX::CreateDDSTextureFromFile(device, deviceContext, std::wstring(fileName.begin(), fileName.end()).c_str(), &resource, &shaderResourceView);
				D3D11_SAMPLER_DESC samplerDescription;
				ZeroMemory(&samplerDescription, sizeof(D3D11_SAMPLER_DESC));
				samplerDescription.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
				samplerDescription.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
				samplerDescription.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
				samplerDescription.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
				samplerDescription.MaxAnisotropy = 1;
				samplerDescription.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
				samplerDescription.MaxLOD = D3D11_FLOAT32_MAX;
				PGA_Rendering_D3D_checkedCall(device->CreateSamplerState(&samplerDescription, &sampler));
			}

			Texture::Texture(const Texture&& other)
			{
				resource = other.resource;
				shaderResourceView = other.shaderResourceView;
				sampler = other.sampler;
				if (resource)
					resource->AddRef();
				if (shaderResourceView)
					shaderResourceView->AddRef();
				if (sampler)
					sampler->AddRef();
			}

			Texture::~Texture()
			{
				if (resource)
					resource->Release();
				if (sampler)
					sampler->Release();
				if (shaderResourceView)
					shaderResourceView->Release();
			}

			Texture& Texture::operator=(const Texture&& other)
			{
				resource = other.resource;
				shaderResourceView = other.shaderResourceView;
				sampler = other.sampler;
				if (resource)
					resource->AddRef();
				if (shaderResourceView)
					shaderResourceView->AddRef();
				if (sampler)
					sampler->AddRef();
				return *this;
			}

			void Texture::bind(ID3D11DeviceContext* deviceContext, unsigned int slot) const
			{
				ID3D11ShaderResourceView* pShaderResourceView = shaderResourceView;
				deviceContext->PSSetShaderResources(slot, 1, &pShaderResourceView);
				ID3D11SamplerState* pSampler = sampler;
				deviceContext->PSSetSamplers(slot, 1, &pSampler);
			}

		}

	}

}