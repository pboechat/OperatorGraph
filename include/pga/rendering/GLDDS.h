#pragma once

#include <GL/glew.h>
#include <windows.h>

#include <iostream>
#include <string>
#include <vector>

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
