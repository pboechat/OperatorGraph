#pragma once

#include <GL/glew.h>
#include <windows.h>

#include <exception>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class Exception : public std::exception
			{
			public:
				Exception(GLenum errorCode, const char* file, int line);

			};

			inline void checkError(const char* file, int line)
			{
				if (GLenum error = glGetError())
					throw PGA::Rendering::GL::Exception(error, file, line);
			}

		}

	}

}

#define PGA_Rendering_GL_checkError() PGA::Rendering::GL::checkError(__FILE__, __LINE__)