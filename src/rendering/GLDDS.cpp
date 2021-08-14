#include "BaseDDS.h"

#include <pga/rendering/GLDDS.h>
#include <pga/rendering/GLException.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			struct GLFormatInfo
			{
				GLenum format;
				GLenum dataFormat;
				GLenum dataType;
				int pixelSize;
			};

			GLuint createTexture2D(GLsizei width, GLsizei height, GLint levels, GLenum format)
			{
				GLuint texture;
				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_2D, texture);
				glTexStorage2D(GL_TEXTURE_2D, levels, format, width, height);
				PGA_Rendering_GL_checkError();
				return texture;
			}

			GLFormatInfo pixelFormatToGLFormatInfo(const BaseDDS::DDSPixelFormat& pixelFormat)
			{
				GLFormatInfo info;
				if (pixelFormat.dwFlags == BaseDDS::DDPF_RGBA && pixelFormat.dwRGBBitCount == 32)
				{
					info.format = GL_SRGB8_ALPHA8;
					info.pixelSize = 4;
					info.dataType = GL_UNSIGNED_BYTE;
					if (pixelFormat.dwABitMask == 0xFF000000U && pixelFormat.dwBBitMask == 0x00FF0000U && pixelFormat.dwGBitMask == 0x0000FF00U && pixelFormat.dwRBitMask == 0x000000FFU)
					{
						info.dataFormat = GL_RGBA;
						return info;
					}
					if (pixelFormat.dwABitMask == 0xFF000000U && pixelFormat.dwRBitMask == 0x00FF0000U && pixelFormat.dwGBitMask == 0x0000FF00U && pixelFormat.dwBBitMask == 0x000000FFU)
					{
						info.dataFormat = GL_BGRA;
						return info;
					}
				}
				else if (pixelFormat.dwFlags == 0x80000 && pixelFormat.dwRGBBitCount == 32)
				{
					info.format = GL_RG16_SNORM;
					info.pixelSize = 4;
					info.dataType = GL_SHORT;
					info.dataFormat = GL_RG;
					return info;
				}
				else if (pixelFormat.dwFlags == 0x40 && pixelFormat.dwRGBBitCount == 32)
				{
					info.format = GL_RG16;
					info.pixelSize = 4;
					info.dataType = GL_UNSIGNED_SHORT;
					info.dataFormat = GL_RG;
					return info;
				}
				else if (pixelFormat.dwFlags == BaseDDS::DDPF_FOURCC)
				{
					switch (pixelFormat.dwFourCC)
					{
					case 111:
						info.pixelSize = 2;
						info.format = GL_R16F;
						info.dataFormat = GL_RED;
						info.dataType = GL_HALF_FLOAT;
						return info;
					case 114:
						info.pixelSize = 4;
						info.format = GL_R32F;
						info.dataFormat = GL_RED;
						info.dataType = GL_FLOAT;
						return info;
					case 113:
						info.pixelSize = 8;
						info.format = GL_RGBA16F;
						info.dataFormat = GL_RGBA;
						info.dataType = GL_HALF_FLOAT;
						return info;
					case 116:
						info.pixelSize = 16;
						info.format = GL_RGBA32F;
						info.dataFormat = GL_RGBA;
						info.dataType = GL_FLOAT;
						return info;
					}
				}
				throw std::runtime_error("PGA::Rendering::GL::DDS::pixelFormatToGLFormatInfo(): unsupported pixel format");
			}

			void GLtoPixelformat(GLenum format, BaseDDS::DDSPixelFormat& pixelFormat, GLFormatInfo& glFormatInfo)
			{
				pixelFormat.dwSize = sizeof(pixelFormat);
				glFormatInfo.format = format;
				switch (format)
				{
				case GL_R16F:
					pixelFormat.dwFlags = BaseDDS::DDPF_FOURCC;
					pixelFormat.dwFourCC = 111;
					glFormatInfo.dataType = GL_HALF_FLOAT;
					glFormatInfo.dataFormat = GL_RED;
					glFormatInfo.pixelSize = 2;
					return;

				case GL_R32F:
					pixelFormat.dwFlags = BaseDDS::DDPF_FOURCC;
					pixelFormat.dwFourCC = 114;
					glFormatInfo.dataType = GL_FLOAT;
					glFormatInfo.dataFormat = GL_RED;
					glFormatInfo.pixelSize = 4;
					return;

				case GL_RGBA8:
				case GL_SRGB8_ALPHA8:
					pixelFormat.dwFlags = BaseDDS::DDPF_RGBA;
					pixelFormat.dwRGBBitCount = 32;
					pixelFormat.dwABitMask = 0xFF000000U;
					pixelFormat.dwRBitMask = 0x00FF0000U;
					pixelFormat.dwGBitMask = 0x0000FF00U;
					pixelFormat.dwBBitMask = 0x000000FFU;
					glFormatInfo.dataType = GL_UNSIGNED_BYTE;
					glFormatInfo.dataFormat = GL_BGRA;
					glFormatInfo.pixelSize = 4;
					return;

				case GL_RGB8:
				case GL_SRGB8:
					pixelFormat.dwFlags = BaseDDS::DDPF_RGB;
					pixelFormat.dwRGBBitCount = 24;
					pixelFormat.dwABitMask = 0x00000000U;
					pixelFormat.dwRBitMask = 0x00FF0000U;
					pixelFormat.dwGBitMask = 0x0000FF00U;
					pixelFormat.dwBBitMask = 0x000000FFU;
					glFormatInfo.dataType = GL_UNSIGNED_BYTE;
					glFormatInfo.dataFormat = GL_BGR;
					glFormatInfo.pixelSize = 3;
					return;

				case GL_RGBA16F:
					pixelFormat.dwFlags = BaseDDS::DDPF_FOURCC;
					pixelFormat.dwFourCC = 113;
					glFormatInfo.dataType = GL_HALF_FLOAT;
					glFormatInfo.dataFormat = GL_RGBA;
					glFormatInfo.pixelSize = 8;
					return;

				case GL_RGBA32F:
					pixelFormat.dwFlags = BaseDDS::DDPF_FOURCC;
					pixelFormat.dwFourCC = 116;
					glFormatInfo.dataType = GL_FLOAT;
					glFormatInfo.dataFormat = GL_RGBA;
					glFormatInfo.pixelSize = 16;
					return;
				}
				throw std::runtime_error("PGA::Rendering::GL::DDS::GLtoPixelformat(): unsupported pixel format");
			}

			GLuint DDS::readTexture1D(std::istream& input)
			{
				if (!isDDS(input))
					throw std::runtime_error("PGA::Rendering::GL::DDS::readTexture1D(): input not a dds stream");
				BaseDDS::DDSHeader header;
				input.read(reinterpret_cast<char*>(&header), sizeof(header));
				GLFormatInfo info = pixelFormatToGLFormatInfo(header.ddpf);
				std::unique_ptr<char[]> buffer(new char[header.dwWidth * info.pixelSize]);
				GLuint texture;
				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_1D, texture);
				int levels = header.dwMipMapCount == 0 ? 1 : header.dwMipMapCount;
				int width = header.dwWidth;
				for (int level = 0; level < levels; ++level)
				{
					BaseDDS::readSurface(input, buffer.get(), width, 4);
					glTexImage1D(GL_TEXTURE_1D, level, info.format, width, 0, info.dataFormat, info.dataType, buffer.get());
					PGA_Rendering_GL_checkError();
					width = width >> 1;
					if (width < 1)
						width = 1;
				}
				PGA_Rendering_GL_checkError();
				return texture;
			}

			bool DDS::isDDS(std::istream& input)
			{
				return BaseDDS::isDDS(input);
			}

			bool DDS::isDDS(const std::string& fileName)
			{
				std::ifstream file(fileName.c_str(), std::ios::in | std::ios::binary);
				if (!file)
					throw std::runtime_error("PGA::Rendering::GL::DDS::isDDS(..): error opening file '" + fileName + '\'');
				return BaseDDS::isDDS(file);
			}

			GLuint DDS::readTexture2D(std::istream& input)
			{
				if (!isDDS(input))
					throw std::runtime_error("PGA::Rendering::GL::DDS::readTexture2D(): input not a dds stream");
				BaseDDS::DDSHeader header;
				input.read(reinterpret_cast<char*>(&header), sizeof(header));
				GLFormatInfo info = pixelFormatToGLFormatInfo(header.ddpf);
				int levels = header.dwMipMapCount == 0 ? 1 : header.dwMipMapCount;
				int width = header.dwWidth;
				int height = header.dwHeight;
				std::unique_ptr<char[]> buffer(new char[width * height * info.pixelSize]);
				GLuint texture = createTexture2D(width, height, levels, info.format);
				for (int level = 0; level < levels; ++level)
				{
					BaseDDS::readSurface(input, buffer.get(), width, height, info.pixelSize);
					glTexSubImage2D(GL_TEXTURE_2D, level, 0, 0, width, height, info.dataFormat, info.dataType, buffer.get());
					PGA_Rendering_GL_checkError();
					width = width >> 1;
					if (width < 1)
						width = 1;
					height = height >> 1;
					if (height < 1)
						height = 1;
				}
				PGA_Rendering_GL_checkError();
				return texture;
			}

			std::ostream& DDS::writeTexture2D(std::ostream& out, GLuint texture)
			{
				BaseDDS::DDSHeader header;
				header.dwSize = sizeof(header);
				header.dwFlags = BaseDDS::DDSD_CAPS | BaseDDS::DDSD_HEIGHT | BaseDDS::DDSD_WIDTH | BaseDDS::DDSD_PITCH | BaseDDS::DDSD_MIPMAPCOUNT | BaseDDS::DDSD_PIXELFORMAT;
				header.dwLinearSize = 0;
				header.dwDepth = 1;
				header.dwMipMapCount = 1;
				std::fill(std::begin(header.dwReserved1), std::end(header.dwReserved1), 0);
				header.dwCaps = BaseDDS::DDSCAPS_TEXTURE;
				header.dwCaps2 = 0;
				header.dwCaps3 = 0;
				header.dwCaps4 = 0;
				header.dwReserved2 = 0;
				glBindTexture(GL_TEXTURE_2D, texture);
				GLint levels = 1;
				GLint format;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
				GLint width;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
				header.dwWidth = width;
				GLint height;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
				header.dwHeight = height;
				GLFormatInfo format_info;
				GLtoPixelformat(format, header.ddpf, format_info);
				header.dwLinearSize = width * format_info.pixelSize;
				std::unique_ptr<char[]> data(new char[width * height * format_info.pixelSize]);
				glGetTexImage(GL_TEXTURE_2D, 0, format_info.dataFormat, format_info.dataType, data.get());
				out.put('D').put('D').put('S').put(' ');
				out.write(reinterpret_cast<const char*>(&header), sizeof(header));
				BaseDDS::writeSurface(out, data.get(), width, height, format_info.pixelSize);
				PGA_Rendering_GL_checkError();
				return out;
			}

		}

	}

}