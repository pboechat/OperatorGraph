#include <pga/rendering/GLMaterial.h>
#include <pga/rendering/GLShader.h>

#include <iterator>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			Material::Material() : shader(nullptr), dirty(false), backFaceCulling(true), blend(false, TRANSPARENCY)
			{
			}

			Material::Material(std::unique_ptr<PGA::Rendering::GL::Shader>&& shader) : shader(std::move(shader)), dirty(false), backFaceCulling(true), blend(false, TRANSPARENCY)
			{
				this->shader->setDefaultParameters(*this);
			}

			Material::~Material()
			{
				textures.clear();
			}

			void Material::setShader(std::unique_ptr<PGA::Rendering::GL::Shader>&& shader)
			{
				this->shader = std::move(shader);
				this->shader->setDefaultParameters(*this);
			}

			void Material::activate(unsigned int pass, bool updateOnlyIfDirty)
			{
				backFaceCulling.set();
				blend.set();
				if (shader == nullptr)
					return;
				shader->bind(pass);
				if (!updateOnlyIfDirty || dirty)
				{
					for (auto& value : floats)
						value.second.second = shader->getUniformLocation(value.first);
					for (auto& value : float2s)
						value.second.second = shader->getUniformLocation(value.first);
					for (auto& value : float3s)
						value.second.second = shader->getUniformLocation(value.first);
					for (auto& value : float4s)
						value.second.second = shader->getUniformLocation(value.first);
					for (auto& value : float4x4s)
						value.second.second = shader->getUniformLocation(value.first);
					for (auto& value : textures)
						value.second.second = shader->getUniformLocation(value.first);
					dirty = false;
				}
				for (auto& value : floats)
					shader->setFloat(value.second.second, value.second.first);
				for (auto& value : float2s)
					shader->setFloat2(value.second.second, value.second.first);
				for (auto& value : float3s)
					shader->setFloat3(value.second.second, value.second.first);
				for (auto& value : float4s)
					shader->setFloat4(value.second.second, value.second.first);
				for (auto& value : float4x4s)
					shader->setFloat4x4(value.second.second, value.second.first);
				unsigned int textureUnit = 0;
				for (auto& value : textures)
				{
					value.second.first->activate(textureUnit);
					shader->setTextureUnit(value.second.second, textureUnit);
					textureUnit++;
				}
			}

			void Material::deactivate() const
			{
				backFaceCulling.restore();
				blend.restore();
			}

			void Material::setParameter(const std::string& name, float value)
			{
				floats[name] = { value, 0u };
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float2& value)
			{
				float2s[name] = { value, 0u };
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float3& value)
			{
				float3s[name] = { value, 0u };
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float4& value)
			{
				float4s[name] = { value, 0u };
				dirty = true;
			}

			void Material::setParameter(const std::string& name, const math::float4x4& value)
			{
				float4x4s[name] = { value, 0u };
				dirty = true;
			}

			void Material::setParameter(const std::string& name, std::unique_ptr<Texture1D>&& value)
			{
				textures.emplace(name, std::make_pair(std::move(value), 0u));
				dirty = true;
			}

			void Material::setParameter(const std::string& name, Texture1D&& value)
			{
				textures.emplace(name, std::make_pair(std::unique_ptr<PGA::Rendering::GL::Texture>(new Texture1D(std::move(value))), 0u));
				dirty = true;
			}

			void Material::setParameter(const std::string& name, std::unique_ptr<Texture2D>&& value)
			{
				textures.emplace(name, std::make_pair(std::move(value), 0u));
				dirty = true;
			}

			void Material::setParameter(const std::string& name, Texture2D&& value)
			{
				textures.emplace(name, std::make_pair(std::unique_ptr<PGA::Rendering::GL::Texture>(new Texture2D(std::move(value))), 0u));
				dirty = true;
			}

			float Material::getFloatParameter(const std::string& name) const
			{
				return floats.find(name)->second.first;
			}

			math::float2 Material::getFloat2Parameter(const std::string& name) const
			{
				return float2s.find(name)->second.first;
			}

			math::float3 Material::getFloat3Parameter(const std::string& name) const
			{
				return float3s.find(name)->second.first;
			}

			math::float4 Material::getFloat4Parameter(const std::string& name) const
			{
				return float4s.find(name)->second.first;
			}

			math::float4x4 Material::getFloat4x4Parameter(const std::string& name) const
			{
				return float4x4s.find(name)->second.first;
			}

			const std::unique_ptr<PGA::Rendering::GL::Texture>& Material::getTextureParameter(const std::string& name) const
			{
				return textures.find(name)->second.first;
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

			bool Material::getBackFaceCulling() const
			{
				return backFaceCulling;
			}

			void Material::setBackFaceCulling(bool backFaceCulling)
			{
				this->backFaceCulling = backFaceCulling;
			}

			BlendMode Material::getBlendMode() const
			{
				return blend.getMode();
			}

			void Material::setBlendMode(BlendMode blendMode)
			{
				blend.setMode(blendMode);
			}

			const std::unique_ptr<PGA::Rendering::GL::Shader>& Material::getShader() const
			{
				return shader;
			}

			std::unique_ptr<PGA::Rendering::GL::Shader>& Material::getShader()
			{
				return shader;
			}

			bool Material::getBlend() const
			{
				return blend;
			}

			void Material::setBlend(bool blend)
			{
				this->blend = blend;
			}

		}

	}

}
