#pragma once

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>

#include "Image.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class Texture
			{
			protected:
				GLuint texture;
				GLuint sampler;

				virtual void updateToDeviceMemory(Image<unsigned int>&& source) = 0;

				Texture();
				Texture(GLuint texture);
				Texture(Texture&& other);
				Texture& operator=(Texture&& other);

			public:
				virtual ~Texture();
				Texture(const Texture&) = delete;
				Texture& operator=(const Texture&) = delete;

				GLuint getTextureRef() const;
				virtual void activate(unsigned int textureUnit) = 0;

			};

			class Texture1D : public Texture
			{
			private:
				virtual void updateToDeviceMemory(Image<unsigned int>&& source);
				bool useFiltering;

			public:
				Texture1D(GLuint texture, bool useFiltering = false);
				Texture1D(Texture1D&& other);
				Texture1D(Image<unsigned int>& source);
				Texture1D(Image<unsigned int>&& source);
				Texture1D& operator=(Texture1D&& other);

				Texture1D(const Texture1D&) = delete;
				Texture1D& operator=(const Texture1D&) = delete;

				virtual void activate(unsigned int textureUnit);

			};

			class Texture2D : public Texture
			{
			private:
				virtual void updateToDeviceMemory(Image<unsigned int>&& source);
				bool useMipMaps;

			public:
				Texture2D(GLuint texture, bool useMipMaps = false);
				Texture2D(Texture2D&& other);
				Texture2D(Image<unsigned int>& source);
				Texture2D(Image<unsigned int>&& source);
				Texture2D& operator=(Texture2D&& other);

				Texture2D(const Texture2D&) = delete;
				Texture2D& operator=(const Texture2D&) = delete;

				virtual void activate(unsigned int textureUnit);

			};

		}

	}

}