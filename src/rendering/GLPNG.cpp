#include <math/math.h>
#include <pga/rendering/GLException.h>
#include <pga/rendering/GLPNG.h>
#include <png.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			Image<unsigned int> PNG::loadFromDisk(const std::string& fileName)
			{
				std::FILE* file;
				png_structp pngReadStructPtr;
				png_infop pngInfoPtr;
				file = std::fopen(fileName.c_str(), "rb");
				if (file == nullptr)
					throw std::runtime_error("PGA::Rendering::PNG::GL::loadFromDisk(): unable to open file [fileName=\"" + fileName + "\"]");
				pngReadStructPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
				if (pngReadStructPtr == nullptr)
				{
					std::fclose(file);
					throw std::runtime_error("PGA::Rendering::PNG::GL::loadFromDisk(): png_create_read_struct() failed");
				}
				pngInfoPtr = png_create_info_struct(pngReadStructPtr);
				if (pngInfoPtr == nullptr)
				{
					std::fclose(file);
					png_destroy_read_struct(&pngReadStructPtr, nullptr, nullptr);
					throw std::runtime_error("PGA::Rendering::PNG::GL::loadFromDisk(): png_create_info_struct() failed");
				}
				if (setjmp(png_jmpbuf(pngReadStructPtr)))
				{
					std::fclose(file);
					png_destroy_read_struct(&pngReadStructPtr, &pngInfoPtr, nullptr);
					throw std::runtime_error("PGA::Rendering::PNG::GL::loadFromDisk(): error reading from file [fileName=\"" + fileName + "\"]");
				}
				png_init_io(pngReadStructPtr, file);
				png_read_info(pngReadStructPtr, pngInfoPtr);
				png_uint_32 width, height;
				int bitDepth, colorType, interlaceMethod, compressionMethod, filterMethod;
				png_get_IHDR(pngReadStructPtr, pngInfoPtr, &width, &height, &bitDepth, &colorType, &interlaceMethod, &compressionMethod, &filterMethod);
				if (bitDepth > 8)
					png_set_scale_16(pngReadStructPtr);
				else if (bitDepth < 8)
					png_set_packing(pngReadStructPtr);
				if ((colorType | PNG_COLOR_MASK_ALPHA) != 0)
					png_set_add_alpha(pngReadStructPtr, 0xFF, PNG_FILLER_AFTER);
				else if (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA)
				{
					png_set_gray_to_rgb(pngReadStructPtr);
					png_set_add_alpha(pngReadStructPtr, 0xFF, PNG_FILLER_AFTER);
				}
				else if (colorType = PNG_COLOR_TYPE_PALETTE)
				{
					png_set_palette_to_rgb(pngReadStructPtr);
					png_set_add_alpha(pngReadStructPtr, 0xFF, PNG_FILLER_AFTER);
				}
				else
				{
					std::fclose(file);
					png_destroy_read_struct(&pngReadStructPtr, &pngInfoPtr, nullptr);
					throw std::runtime_error("PGA::Rendering::PNG::GL::loadFromDisk(): unsupported color type [colorType=" + std::to_string(colorType) + ", fileName=\"" + fileName + "\"]");
				}
				png_read_update_info(pngReadStructPtr, pngInfoPtr);
				Image<unsigned int> image(width, height);
				std::unique_ptr<png_byte*[]> rows(new png_byte*[height]);
				for (size_t y = 0; y < height; ++y)
					rows[y] = reinterpret_cast<png_byte*>(rawData(image) + (height - y - 1) * width);
				png_read_image(pngReadStructPtr, &rows[0]);
				png_read_end(pngReadStructPtr, pngInfoPtr);
				png_destroy_read_struct(&pngReadStructPtr, &pngInfoPtr, nullptr);
				std::fclose(file);
				return image;
			}

			void PNG::writeToDisk(const std::string& fileName, unsigned int* data, unsigned int width, unsigned int height)
			{
				std::FILE* file;
				png_structp pngWriteStructPtr;
				png_infop pngInfoPtr;
				file = std::fopen(fileName.c_str(), "wb");
				if (file == nullptr)
					throw std::runtime_error("PGA::Rendering::PNG::GL::writeDataToDisk(): unable to open file [fileName=\"" + fileName + "\"]");
				pngWriteStructPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
				if (pngWriteStructPtr == nullptr)
				{
					std::fclose(file);
					throw std::runtime_error("PGA::Rendering::PNG::GL::writeDataToDisk(): png_create_write_struct() failed");
				}
				pngInfoPtr = png_create_info_struct(pngWriteStructPtr);
				if (pngInfoPtr == nullptr)
				{
					std::fclose(file);
					png_destroy_write_struct(&pngWriteStructPtr, nullptr);
					throw std::runtime_error("PGA::Rendering::PNG::GL::writeDataToDisk(): png_create_info_struct() failed");
				}
				if (setjmp(png_jmpbuf(pngWriteStructPtr)))
				{
					std::fclose(file);
					png_destroy_write_struct(&pngWriteStructPtr, &pngInfoPtr);
					throw std::runtime_error("PGA::Rendering::PNG::GL::writeDataToDisk(): error writing to file [fileName=\"" + fileName + "\"]");
				}
				png_init_io(pngWriteStructPtr, file);
				png_set_IHDR(pngWriteStructPtr, pngInfoPtr, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
				png_write_info(pngWriteStructPtr, pngInfoPtr);
				std::unique_ptr<png_byte*[]> rows(new png_byte*[height]);
				for (unsigned int y = 0; y < height; ++y)
					rows[y] = reinterpret_cast<png_byte*>(data + (height - y - 1) * width);
				png_write_image(pngWriteStructPtr, rows.get());
				png_write_end(pngWriteStructPtr, pngInfoPtr);
				png_destroy_write_struct(&pngWriteStructPtr, &pngInfoPtr);
				std::fclose(file);
			}

			void PNG::writeGLTextureToDisk(const std::string& fileName, GLuint texture)
			{
				glBindTexture(GL_TEXTURE_2D, texture);
				GLint internalFormat;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &internalFormat);
				GLint width, height;
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
				glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
				std::unique_ptr<unsigned int[]> data(new unsigned int[width * height]);
				glGetTexImage(GL_TEXTURE_2D, 0, internalFormat, GL_UNSIGNED_INT, data.get());
				writeToDisk(fileName, data.get(), width, height);
				PGA_Rendering_GL_checkError();
			}

			GLuint PNG::createCubemap(const Image<unsigned int>& right,
				const Image<unsigned int>& left,
				const Image<unsigned int>& top,
				const Image<unsigned int>& bottom,
				const Image<unsigned int>& front,
				const Image<unsigned int>& back)
			{
				GLuint texture;
				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, width(right), height(right), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(right));
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				PGA_Rendering_GL_checkError();
				glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, width(left), height(left), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(left));
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				PGA_Rendering_GL_checkError();
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, width(top), height(top), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(top));
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				PGA_Rendering_GL_checkError();
				glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, width(bottom), height(bottom), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(bottom));
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				PGA_Rendering_GL_checkError();
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, width(front), height(front), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(front));
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				PGA_Rendering_GL_checkError();
				glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, width(back), height(back), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(back));
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
				PGA_Rendering_GL_checkError();
				return texture;
			}

			GLuint PNG::createTexture2D(const Image<unsigned int>& image)
			{
				GLuint texture;
				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_2D, texture);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width(image), height(image), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(image));
				glGenerateMipmap(GL_TEXTURE_2D);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				PGA_Rendering_GL_checkError();
				return texture;
			}

			GLuint PNG::createTexture2D(const Image<unsigned int>& image, bool generateMipMaps)
			{
				GLuint texture;
				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_2D, texture);
				unsigned int numMipMaps;
				if (generateMipMaps)
				{
					unsigned int size;
					if (width(image) < height(image))
						size = width(image);
					else
						size = height(image);
					numMipMaps = static_cast<unsigned int>(floor(log(static_cast<double>(size)) / log(2)));
				}
				else
					numMipMaps = 0;
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width(image), height(image), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(image));
				if (generateMipMaps)
					glGenerateMipmap(GL_TEXTURE_2D);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (generateMipMaps) ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				PGA_Rendering_GL_checkError();
				return texture;
			}

			GLuint PNG::createTexture1D(const Image<unsigned int>& image)
			{
				GLuint texture;
				glGenTextures(1, &texture);
				glBindTexture(GL_TEXTURE_1D, texture);
				glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, width(image) * height(image), 0, GL_RGBA, GL_UNSIGNED_BYTE, rawData(image));
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_REPEAT);
				PGA_Rendering_GL_checkError();
				return texture;
			}

			GLuint PNG::loadCubemap(const std::string& rightFilename,
				const std::string& leftFileName,
				const std::string& topFileName,
				const std::string& bottomFileName,
				const std::string& frontFileName,
				const std::string& backFileName)
			{
				return createCubemap(loadFromDisk(rightFilename),
					loadFromDisk(leftFileName),
					loadFromDisk(topFileName),
					loadFromDisk(bottomFileName),
					loadFromDisk(frontFileName),
					loadFromDisk(backFileName));
			}

			GLuint PNG::loadTexture2D(const std::string& fileName)
			{
				return createTexture2D(loadFromDisk(fileName));
			}

			GLuint PNG::loadTexture2D(const std::string& fileName, bool generateMipMaps)
			{
				return createTexture2D(loadFromDisk(fileName), generateMipMaps);
			}

			GLuint PNG::loadTexture1D(const std::string& fileName)
			{
				return createTexture1D(loadFromDisk(fileName));
			}

		}

	}

}
