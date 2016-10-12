#pragma once

#include <string>
#include <vector>
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
			class PNG
			{
			public:
				static GLuint createCubemap(const Image<unsigned int>& right, const Image<unsigned int>& left, const Image<unsigned int>& top, const Image<unsigned int>& bottom, const Image<unsigned int>& front, const Image<unsigned int>& back);
				static GLuint createTexture2D(const Image<unsigned int>& image);
				static GLuint createTexture2D(const Image<unsigned int>& image, bool generateMipMaps);
				static GLuint createTexture1D(const Image<unsigned int>& image);
				static GLuint loadTexture1D(const std::string& fileName);
				static GLuint loadTexture2D(const std::string& fileName);
				static GLuint loadTexture2D(const std::string& fileName, bool generateMipMaps);
				static GLuint loadCubemap(const std::string& rightFileName, const std::string& leftFileName, const std::string& topFileName, const std::string& bottomFileName, const std::string& frontFileName, const std::string& backFileName);
				static Image<unsigned int> loadFromDisk(const std::string& fileName);
				static void writeToDisk(const std::string& fileName, unsigned int* data, unsigned int width, unsigned int height);
				static void writeGLTextureToDisk(const std::string& fileName, GLuint texture);

			};

		}

	}

}
