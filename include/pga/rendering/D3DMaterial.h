#pragma once

#include "D3DTexture.h"

#include <d3d11.h>
#include <math/matrix.h>
#include <math/vector.h>

#include <map>
#include <memory>
#include <tuple>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class Shader;

			class Material
			{
			private:
				std::unique_ptr<PGA::Rendering::D3D::Shader> shader;
				std::map<std::string, float> floats;
				std::map<std::string, math::float2> float2s;
				std::map<std::string, math::float3> float3s;
				std::map<std::string, math::float4> float4s;
				std::map<std::string, math::float4x4> float4x4s;
				std::map<std::string, std::unique_ptr<PGA::Rendering::D3D::Texture>> textures;
				ID3D11Buffer* parametersBuffer;
				std::unique_ptr<void, decltype(free)*> parametersBufferData;
				bool dirty;

			public:
				Material();
				virtual ~Material();
				Material(const Material&) = delete;
				Material& operator=(Material&) = delete;
				Material(std::unique_ptr<PGA::Rendering::D3D::Shader>&& shader);

				void setShader(std::unique_ptr<PGA::Rendering::D3D::Shader>&& shader);
				void setParameter(const std::string& name, float value);
				void setParameter(const std::string& name, const math::float2& value);
				void setParameter(const std::string& name, const math::float3& value);
				void setParameter(const std::string& name, const math::float4& value);
				void setParameter(const std::string& name, const math::float4x4& value);
				void setParameter(const std::string& name, std::unique_ptr<PGA::Rendering::D3D::Texture>&& value);
				void setParameter(const std::string& name, PGA::Rendering::D3D::Texture&& value);
				const std::unique_ptr<PGA::Rendering::D3D::Shader>& getShader() const;
				std::unique_ptr<PGA::Rendering::D3D::Shader>& getShader();
				float getFloatParameter(const std::string& name) const;
				math::float2 getFloat2Parameter(const std::string& name) const;
				math::float3 getFloat3Parameter(const std::string& name) const;
				math::float4 getFloat4Parameter(const std::string& name) const;
				math::float4x4 getFloat4x4Parameter(const std::string& name) const;
				const std::unique_ptr<PGA::Rendering::D3D::Texture>& getTextureParameter(const std::string& name) const;
				bool hasFloatParameter(const std::string& name) const;
				bool hasFloat2Parameter(const std::string& name) const;
				bool hasFloat3Parameter(const std::string& name) const;
				bool hasFloat4Parameter(const std::string& name) const;
				bool hasFloat4x4Parameter(const std::string& name) const;
				bool hasTextureParameter(const std::string& name) const;
				void prepare(ID3D11Device* device);
				void activate(ID3D11DeviceContext* deviceContext, unsigned int parameterBufferSlot = 2);
				void deactivate() const;

			};

		}

	}

}
