#include <string>

#include <pga/rendering/GLException.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			std::string getErrorMessage(GLenum errorCode, const char* file, int line)
			{
				std::string errorStr;
				switch (errorCode)
				{
				case GL_INVALID_ENUM:
					errorStr = "GL_INVALID_ENUM";
				case GL_INVALID_VALUE:
					errorStr = "GL_INVALID_VALUE";
				case GL_INVALID_OPERATION:
					errorStr = "GL_INVALID_OPERATION";
				case GL_INVALID_FRAMEBUFFER_OPERATION:
					errorStr = "GL_INVALID_FRAMEBUFFER_OPERATION";
				default:
					errorStr = "unknown error code";
				}
				return std::string(file) +
					'(' + std::to_string(static_cast<long long>(line)) + "): " + errorStr;
			}

			Exception::Exception(GLenum errorCode, const char* file, int line) : std::exception(getErrorMessage(errorCode, file, line).c_str())
			{
			}

		}

	}

}