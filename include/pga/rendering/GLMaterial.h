#pragma once

#include "GLTexture.h"

#include <math/matrix.h>
#include <math/vector.h>

#include <map>
#include <memory>
#include <tuple>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class Shader;

			enum BlendMode
			{
				TRANSPARENCY,
				MULTIPLICATIVE,
				ADDITIVE

			};

			template <GLenum StateT>
			class RenderState
			{
			private:
				bool current;
				bool previous;

			protected:
				virtual void onEnable() {}
				virtual void onDisable() {}
				virtual void onRestore() const {}

			public:
				RenderState(bool current) : current(current) {}
				bool operator = (bool value)
				{
					return (current = value);
				}
				bool operator == (bool value) const
				{
					return (current == value);
				}
				bool operator != (bool value) const
				{
					return !operator==(value);
				}
				operator bool() const
				{
					return current;
				}
				void set()
				{
					previous = glIsEnabled(StateT) != 0;
					if (previous == current) return;
					if (current)
					{
						onEnable();
						glEnable(StateT);
					}
					else
					{
						onDisable();
						glDisable(StateT);
					}
				}
				void restore() const
				{
					if (previous == current) return;
					if (previous)
					{
						onRestore();
						glEnable(StateT);
					}
					else
					{
						glDisable(StateT);
					}
				}

			};

			class Blend : public RenderState < GL_BLEND >
			{
			private:
				GLenum getSourceFactor()
				{
					switch (mode)
					{
					case PGA::Rendering::GL::BlendMode::TRANSPARENCY:
						return GL_SRC_ALPHA;
					case PGA::Rendering::GL::BlendMode::MULTIPLICATIVE:
						return GL_ZERO;
					case PGA::Rendering::GL::BlendMode::ADDITIVE:
						return GL_ONE;
					default:
						throw std::runtime_error("PGA::Rendering::GL::Material::Blend::getSourceFactor(): unknown blend mode");
					}
				}

				GLenum getDestinationFactor()
				{
					switch (mode)
					{
					case PGA::Rendering::GL::BlendMode::TRANSPARENCY:
						return GL_ONE_MINUS_SRC_ALPHA;
					case PGA::Rendering::GL::BlendMode::MULTIPLICATIVE:
						return GL_SRC_COLOR;
					case PGA::Rendering::GL::BlendMode::ADDITIVE:
						return GL_ONE;
					default:
						throw std::runtime_error("PGA::Rendering::GL::Material::Blend::getSourceFactor(): unknown blend mode");
					}
				}

			protected:
				BlendMode mode;

				virtual void onEnable()
				{
					glBlendFunc(getSourceFactor(), getDestinationFactor());
				}

				virtual void onDisable()
				{
					// TODO: pool current blend func
				}

				virtual void onRestore() const
				{
					// TODO: restore previous blend func
				}


			public:
				inline BlendMode getMode() const
				{
					return mode;
				}

				inline void setMode(BlendMode mode)
				{
					this->mode = mode;
				}

				Blend(bool current, BlendMode mode) : RenderState(current), mode(mode) {}
				using RenderState<GL_BLEND>::operator=;
				using RenderState<GL_BLEND>::operator==;
				using RenderState<GL_BLEND>::operator!=;

			};

			class Material
			{
			private:
				std::unique_ptr<PGA::Rendering::GL::Shader> shader;
				std::map<std::string, std::pair<float, unsigned int>> floats;
				std::map<std::string, std::pair<math::float2, unsigned int>> float2s;
				std::map<std::string, std::pair<math::float3, unsigned int>> float3s;
				std::map<std::string, std::pair<math::float4, unsigned int>> float4s;
				std::map<std::string, std::pair<math::float4x4, unsigned int>> float4x4s;
				std::map<std::string, std::pair<std::unique_ptr<PGA::Rendering::GL::Texture>, unsigned int>> textures;
				RenderState<GL_CULL_FACE> backFaceCulling;
				Blend blend;
				bool dirty;

			public:
				Material();
				Material(const Material&) = delete;
				Material& operator=(Material&) = delete;
				Material(std::unique_ptr<PGA::Rendering::GL::Shader>&& shader);
				virtual ~Material();

				bool getBackFaceCulling() const;
				void setBackFaceCulling(bool backFaceCulling);
				bool getBlend() const;
				void setBlend(bool blend);
				BlendMode getBlendMode() const;
				void setBlendMode(BlendMode blendMode);
				void setShader(std::unique_ptr<PGA::Rendering::GL::Shader>&& shader);
				void setParameter(const std::string& name, float value);
				void setParameter(const std::string& name, const math::float2& value);
				void setParameter(const std::string& name, const math::float3& value);
				void setParameter(const std::string& name, const math::float4& value);
				void setParameter(const std::string& name, const math::float4x4& value);
				void setParameter(const std::string& name, std::unique_ptr<Texture1D>&& value);
				void setParameter(const std::string& name, Texture1D&& value);
				void setParameter(const std::string& name, std::unique_ptr<Texture2D>&& value);
				void setParameter(const std::string& name, Texture2D&& value);
				const std::unique_ptr<PGA::Rendering::GL::Shader>& getShader() const;
				std::unique_ptr<PGA::Rendering::GL::Shader>& getShader();
				float getFloatParameter(const std::string& name) const;
				math::float2 getFloat2Parameter(const std::string& name) const;
				math::float3 getFloat3Parameter(const std::string& name) const;
				math::float4 getFloat4Parameter(const std::string& name) const;
				math::float4x4 getFloat4x4Parameter(const std::string& name) const;
				const std::unique_ptr<PGA::Rendering::GL::Texture>& getTextureParameter(const std::string& name) const;
				bool hasFloatParameter(const std::string& name) const;
				bool hasFloat2Parameter(const std::string& name) const;
				bool hasFloat3Parameter(const std::string& name) const;
				bool hasFloat4Parameter(const std::string& name) const;
				bool hasFloat4x4Parameter(const std::string& name) const;
				bool hasTextureParameter(const std::string& name) const;
				void activate(unsigned int pass, bool updateOnlyIfDirty = true);
				void deactivate() const;

			};

		}

	}

}
