#include <pga/rendering/GLException.h>
#include <pga/rendering/GLTexture.h>

#include <memory>
#include <string>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			Texture::Texture() : texture(0), sampler(0)
			{
			}

			Texture::Texture(Texture&& other) : texture(0), sampler(0)
			{
				std::swap(texture, other.texture);
				std::swap(sampler, other.sampler);
			}

			Texture::Texture(GLuint texture) : texture(texture), sampler(0)
			{
			}

			Texture::~Texture()
			{
				if (texture)
					glDeleteTextures(1, &texture);
				if (sampler)
					glGenSamplers(1, &sampler);
				PGA_Rendering_GL_checkError();
			}

			Texture& Texture::operator=(Texture&& other)
			{
				std::swap(texture, other.texture);
				std::swap(sampler, other.sampler);
				return *this;
			}

			GLuint Texture::getTextureRef() const
			{
				return texture;
			}

			void Texture2D::updateToDeviceMemory(Image<unsigned int>&& source)
			{
				if (texture)
					glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_2D, texture);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width(source), height(source), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(source));
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, ((useMipMaps) ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR));
				if (sampler)
					glGenSamplers(1, &sampler);
				glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, ((useMipMaps) ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR));
				PGA_Rendering_GL_checkError();
			}

			void Texture2D::activate(unsigned int textureUnit)
			{
				glActiveTexture(GL_TEXTURE0 + textureUnit);
				glBindTexture(GL_TEXTURE_2D, texture);
				glBindSampler(textureUnit, sampler);
			}

			Texture2D::Texture2D(Texture2D&& other) : Texture(std::move(other)), useMipMaps(false)
			{
			}

			Texture2D::Texture2D(Image<unsigned int>& source) : useMipMaps(false)
			{
				updateToDeviceMemory(std::move(source));
			}

			Texture2D::Texture2D(Image<unsigned int>&& source) : useMipMaps(false)
			{
				updateToDeviceMemory(std::move(source));
			}

			Texture2D::Texture2D(GLuint texture, bool useMipMaps) : Texture(texture), useMipMaps(useMipMaps)
			{
			}

			Texture2D& Texture2D::operator=(Texture2D&& other)
			{
				return static_cast<Texture2D&>(Texture::operator=(std::move(other)));
			}

			void Texture1D::updateToDeviceMemory(Image<unsigned int>&& source)
			{
				if (texture)
					glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_1D, texture);
				glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, width(source) * height(source), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(source));
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, (useFiltering) ? GL_LINEAR : GL_NEAREST);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, (useFiltering) ? GL_LINEAR : GL_NEAREST);
				if (sampler)
					glGenSamplers(1, &sampler);
				glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
				glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, (useFiltering) ? GL_LINEAR : GL_NEAREST);
				glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, (useFiltering) ? GL_LINEAR : GL_NEAREST);
				PGA_Rendering_GL_checkError();
			}

			void Texture1D::activate(unsigned int textureUnit)
			{
				glActiveTexture(GL_TEXTURE0 + textureUnit);
				glBindTexture(GL_TEXTURE_1D, texture);
				glBindSampler(textureUnit, sampler);
			}

			Texture1D::Texture1D(Texture1D&& other) : Texture(std::move(other)), useFiltering(false)
			{
			}

			Texture1D::Texture1D(Image<unsigned int>& source) : useFiltering(false)
			{
				updateToDeviceMemory(std::move(source));
			}

			Texture1D::Texture1D(Image<unsigned int>&& source) : useFiltering(false)
			{
				updateToDeviceMemory(std::move(source));
			}

			Texture1D::Texture1D(GLuint texture, bool useFiltering) : Texture(texture), useFiltering(useFiltering)
			{
			}

			Texture1D& Texture1D::operator=(Texture1D&& other)
			{
				return static_cast<Texture1D&>(Texture::operator=(std::move(other)));
			}

		}

	}

}