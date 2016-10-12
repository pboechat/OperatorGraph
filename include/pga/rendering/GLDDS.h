#pragma once

#include <string>
#include <vector>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class DDS
			{
			public:
				static bool isDDS(std::istream& input);
				static bool isDDS(const std::string& fileName);
				static GLuint readTexture1D(std::istream& input);
				static GLuint readTexture2D(std::istream& input);
				static std::ostream& writeTexture2D(std::ostream& input, GLuint texture);

			};

		}

	}

}
