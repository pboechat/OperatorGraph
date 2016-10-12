#include <iterator>

#include <pga/rendering/D3DException.h>
#include <pga/rendering/D3DShader.h>
#include <pga/rendering/D3DMaterial.h>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			void move(void** ppBuffer, size_t offset)
			{
				auto cBuffer = reinterpret_cast<unsigned char*>(*ppBuffer);
				cBuffer += offset;
				(*ppBuffer) = reinterpret_cast<void*>(cBuffer);
			}

			template <size_t N>
			size_t roundUp(size_t n)
			{
				return ((n + (N - 1)) / N) * N;
			}

			template <size_t S, typename T>
			size_t pack16(T data, size_t nBytes, void** ppBuffer = nullptr)
			{
				// 4-byte pack
				auto b = roundUp<4>(S);
				auto c = 16 - (nBytes % 16);
				size_t i, offset;
				// 16-byte boundary
				if (c >= b)
					i = b, offset = 0;
				else
					i = c + b, offset = c;
				if (ppBuffer != nullptr)
				{
					move(ppBuffer, offset);
					//(*reinterpret_cast<T*>(*ppBuffer)) = data;
					memcpy(*ppBuffer, &data, S);
					move(ppBuffer, b);
				}
				return i;
			}

			Material::Material(std::unique_ptr<PGA::Rendering::D3D::Shader>&& shader) : shader(std::move(shader)), dirty(false), parametersBufferData({ nullptr, free }), parametersBuffer(0)
			{
				if (this->shader)
					this->shader->setDefaultParameters(*this);
			}

			Material::Material() : Material(nullptr)
			{
			}

			Material::~Material()
			{
				textures.clear();
				if (parametersBuffer)
					parametersBuffer->Release();
			}

			void Material::setShader(std::unique_ptr<PGA::Rendering::D3D::Shader>&& shader)
			{
				this->shader = std::move(shader);
				this->shader->setDefaultParameters(*this);
			}

			void Material::activate(ID3D11DeviceContext* deviceContext, unsigned int parameterBufferSlot)
			{
				if (dirty)
					throw std::runtime_error("PGA::Rendering::D3D::Material::activate(..): material is unprepared");
				ID3D11Buffer* pConstantBuffer = parametersBuffer;
				deviceContext->VSSetConstantBuffers(parameterBufferSlot, 1, &pConstantBuffer);
				deviceContext->PSSetConstantBuffers(parameterBufferSlot, 1, &pConstantBuffer);
				auto i = 0u;
				for (auto& value : textures)
					value.second->bind(deviceContext, i++);
				if (shader == nullptr)
					return;
				shader->bind(deviceContext);
			}

			void Material::deactivate() const
			{
			}

			void Material::setParameter(const std::string& name, float value)
			{
				floats[name] = value;
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float2& value)
			{
				float2s[name] = value;
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float3& value)
			{
				float3s[name] = value;
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float4& value)
			{
				float4s[name] = value;
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float4x4& value)
			{
				float4x4s[name] = value;
				dirty = true;
			}

			void Material::setParameter(const std::string& name, std::unique_ptr<PGA::Rendering::D3D::Texture>&& value)
			{
				textures.emplace(name, std::move(value));
				dirty = true;
			}

			void Material::setParameter(const std::string& name, PGA::Rendering::D3D::Texture&& value)
			{
				textures.emplace(name, std::unique_ptr<PGA::Rendering::D3D::Texture>(new PGA::Rendering::D3D::Texture(std::move(value))));
				dirty = true;
			}

			float Material::getFloatParameter(const std::string& name) const
			{
				return floats.find(name)->second;
			}

			math::float2 Material::getFloat2Parameter(const std::string& name) const
			{
				return float2s.find(name)->second;
			}

			math::float3 Material::getFloat3Parameter(const std::string& name) const
			{
				return float3s.find(name)->second;
			}

			math::float4 Material::getFloat4Parameter(const std::string& name) const
			{
				return float4s.find(name)->second;
			}

			math::float4x4 Material::getFloat4x4Parameter(const std::string& name) const
			{
				return float4x4s.find(name)->second;
			}

			const std::unique_ptr<PGA::Rendering::D3D::Texture>& Material::getTextureParameter(const std::string& name) const
			{
				return textures.find(name)->second;
			}

			bool Material::hasFloatParameter(const std::string& name) const
			{
				return floats.find(name) != floats.end();
			}

			bool Material::hasFloat2Parameter(const std::string& name) const
			{
				return float2s.find(name) != float2s.end();
			}

			bool Material::hasFloat3Parameter(const std::string& name) const
			{
				return float3s.find(name) != float3s.end();
			}

			bool Material::hasFloat4Parameter(const std::string& name) const
			{
				return float4s.find(name) != float4s.end();
			}

			bool Material::hasFloat4x4Parameter(const std::string& name) const
			{
				return float4x4s.find(name) != float4x4s.end();
			}

			bool Material::hasTextureParameter(const std::string& name) const
			{
				return textures.find(name) != textures.end();
			}

			// NOTE: parameters are packed from the smallest to the biggest type and in alphabetical order
			void Material::prepare(ID3D11Device* device)
			{
				if (!dirty)
					return;
				size_t nBytes = 0;
				for (auto value : floats)
					nBytes += pack16<sizeof(float)>(value.second, nBytes);
				for (auto value : float2s)
					nBytes += pack16<sizeof(float) * 2>(value.second, nBytes);
				for (auto value : float3s)
					nBytes += pack16<sizeof(float) * 3>(value.second, nBytes);
				for (auto value : float4s)
					nBytes += pack16<sizeof(float) * 4>(value.second, nBytes);
				for (auto value : float4x4s)
					nBytes += pack16<sizeof(float) * 16>(value.second, nBytes);
				nBytes = roundUp<16>(nBytes);
				parametersBufferData = std::unique_ptr<void, decltype(free)*>{
					malloc(nBytes),
					free
				};
				void* pBuffer = parametersBufferData.get();
				memset(pBuffer, 0, nBytes);
				for (auto value : floats)
					nBytes += pack16<sizeof(float)>(value.second, nBytes, &pBuffer);
				for (auto value : float2s)
					nBytes += pack16<sizeof(float) * 2>(value.second, nBytes, &pBuffer);
				for (auto value : float3s)
					nBytes += pack16<sizeof(float) * 3>(value.second, nBytes, &pBuffer);
				for (auto value : float4s)
					nBytes += pack16<sizeof(float) * 4>(value.second, nBytes, &pBuffer);
				for (auto value : float4x4s)
					nBytes += pack16<sizeof(float) * 16>(value.second, nBytes, &pBuffer);
				nBytes = roundUp<16>(nBytes);
				D3D11_BUFFER_DESC constantBufferDescription;
				constantBufferDescription.ByteWidth = (UINT)nBytes;
				constantBufferDescription.Usage = D3D11_USAGE_DYNAMIC;
				constantBufferDescription.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
				constantBufferDescription.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				constantBufferDescription.MiscFlags = 0;
				constantBufferDescription.StructureByteStride = 0;
				D3D11_SUBRESOURCE_DATA subResourceData;
				subResourceData.pSysMem = parametersBufferData.get();
				subResourceData.SysMemPitch = 0;
				subResourceData.SysMemSlicePitch = 0;
				PGA_Rendering_D3D_checkedCall(device->CreateBuffer(&constantBufferDescription, &subResourceData, &parametersBuffer));
				dirty = false;
			}

			const std::unique_ptr<PGA::Rendering::D3D::Shader>& Material::getShader() const
			{
				return shader;
			}

			std::unique_ptr<PGA::Rendering::D3D::Shader>& Material::getShader()
			{
				return shader;
			}

		}

	}

}
