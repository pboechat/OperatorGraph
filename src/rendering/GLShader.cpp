#include <vector>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <pga/core/StringUtils.h>
#include <pga/rendering/GLException.h>
#include <pga/rendering/GLShader.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			//////////////////////////////////////////////////////////////////////////
			std::vector<std::string> g_includePaths;

			//////////////////////////////////////////////////////////////////////////
			bool getShaderCompileStatus(GLuint shader)
			{
				GLint status;
				glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
				PGA_Rendering_GL_checkError();
				return status == GL_TRUE;
			}

			//////////////////////////////////////////////////////////////////////////
			bool getProgramLinkStatus(GLuint program)
			{
				GLint status;
				glGetProgramiv(program, GL_LINK_STATUS, &status);
				PGA_Rendering_GL_checkError();
				return status == GL_TRUE;
			}

			//////////////////////////////////////////////////////////////////////////
			std::string getShaderInfoLog(GLuint shader)
			{
				GLint length;
				glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
				PGA_Rendering_GL_checkError();
				if (length == 0)
					return "";
				std::string log(length + 1, '\0');
				glGetShaderInfoLog(shader, length, 0, &log[0]);
				PGA_Rendering_GL_checkError();
				return log;
			}

			//////////////////////////////////////////////////////////////////////////
			std::string getProgramInfoLog(GLuint program)
			{
				GLint length;
				glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
				PGA_Rendering_GL_checkError();
				if (length == 0)
					return "";
				std::string log(length + 1, '\0');
				glGetProgramInfoLog(program, length, 0, &log[0]);
				PGA_Rendering_GL_checkError();
				return log;
			}

			//////////////////////////////////////////////////////////////////////////
			void linkProgram(GLuint program)
			{
				glLinkProgram(program);
				PGA_Rendering_GL_checkError();
				if (getProgramLinkStatus(program) == false)
					throw std::runtime_error(getProgramInfoLog(program));
			}

			//////////////////////////////////////////////////////////////////////////
			void compileShaderSource(GLuint shader, const std::string& source)
			{
				const GLchar* sourceCStr = source.c_str();
				glShaderSource(shader, 1, &sourceCStr, 0);
				glCompileShader(shader);
				PGA_Rendering_GL_checkError();
				if (getShaderCompileStatus(shader) == false)
					throw std::runtime_error(getShaderInfoLog(shader));
			}

			//////////////////////////////////////////////////////////////////////////
			// NOTE: forward declaration
			std::string getShaderSource(const std::string& fileName);

			//////////////////////////////////////////////////////////////////////////
			std::string replaceIncludes(const std::string& shaderSource)
			{
				std::string newShaderSource(shaderSource);
				std::regex includeRE("#include[\\s]*<([^>]+)>");
				std::sregex_iterator next(shaderSource.begin(), shaderSource.end(), includeRE);
				std::sregex_iterator end;
				for (; next != end; next++) 
				{
					std::smatch match = *next;
					auto includeStr = match[0].str();
					auto includeFile = match[1].str();
					PGA::StringUtils::replaceAll(newShaderSource, includeStr, getShaderSource(includeFile));
				}
				return newShaderSource;
			}

			//////////////////////////////////////////////////////////////////////////
			std::string getShaderSource(const std::string& fileName)
			{
				for (auto& path : g_includePaths)
				{
					std::ifstream file(path + fileName);
					if (!file.good())
						continue;
					return replaceIncludes(std::string((std::istreambuf_iterator<char>(file)),
						std::istreambuf_iterator<char>()));
				}
				throw std::runtime_error("PGA::Rendering::GL::getShaderSource(..): shader file couldn't be opened [fileName=" + fileName + "]");
			}

			//////////////////////////////////////////////////////////////////////////
			GLuint compileShader(const std::string& fileName, GLenum type)
			{
				GLuint shader = glCreateShader(type);
				compileShaderSource(shader, getShaderSource(fileName));
				return shader;
			} 

			//////////////////////////////////////////////////////////////////////////
			void Shader::setIncludePath(const std::initializer_list<std::string>& includePaths)
			{
				g_includePaths.clear();
				for (auto& path : includePaths)
				{
					std::string newPath(path);
					if (path.empty())
						continue;
					// NOTE: guarantee that path always ends with a trailing slash (or backslash)
					if (path.back() != '/' && path.back() != '\\')
						g_includePaths.emplace_back(path + '\\');
					else
						g_includePaths.emplace_back(path);
				}
			}

			//////////////////////////////////////////////////////////////////////////
			Shader::Shader(const std::string& vertexShaderFileName, const std::string& fragmentShaderFileName) :
				program(glCreateProgram()),
				vertexShader(compileShader(vertexShaderFileName, GL_VERTEX_SHADER)),
				fragmentShader(compileShader(fragmentShaderFileName, GL_FRAGMENT_SHADER)),
				geometryShader(0)
			{
				glAttachShader(program, vertexShader);
				glAttachShader(program, fragmentShader);
				linkProgram(program);
				PGA_Rendering_GL_checkError();
			}

			Shader::Shader(const std::string& vertexShaderFileName, const std::string& geometryShaderFileName, const std::string& fragmentShaderFileName) :
				program(glCreateProgram()),
				vertexShader(compileShader(vertexShaderFileName, GL_VERTEX_SHADER)),
				geometryShader(compileShader(geometryShaderFileName, GL_GEOMETRY_SHADER)),
				fragmentShader(compileShader(fragmentShaderFileName, GL_FRAGMENT_SHADER))
			{
				glAttachShader(program, vertexShader);
				glAttachShader(program, geometryShader);
				glAttachShader(program, fragmentShader);
				linkProgram(program);
				PGA_Rendering_GL_checkError();
			}

			Shader::Shader(Shader&& other) : vertexShader(std::move(other.vertexShader)), geometryShader(std::move(geometryShader)), fragmentShader(std::move(other.fragmentShader)), program(std::move(program))
			{
			}

			Shader::~Shader()
			{
				if (vertexShader)
					glDeleteShader(vertexShader);
				if (geometryShader)
					glDeleteShader(geometryShader);
				if (fragmentShader)
					glDeleteShader(fragmentShader);
				if (program)
					glDeleteProgram(program);
				PGA_Rendering_GL_checkError();
			}

			GLuint Shader::getUniformLocation(const std::string& name) const
			{
				return glGetUniformLocation(program, name.c_str());
			}

			void Shader::setFloat(const std::string& name, float uniform) const
			{
				glUniform1f(glGetUniformLocation(program, name.c_str()), uniform);
			}

			void Shader::setFloat4(const std::string& name, const math::float4& uniform) const
			{
				glUniform4fv(glGetUniformLocation(program, name.c_str()), 1, &uniform.x);
			}

			void Shader::setFloat3(const std::string& name, const math::float3& uniform) const
			{
				glUniform3fv(glGetUniformLocation(program, name.c_str()), 1, &uniform.x);
			}

			void Shader::setFloat2(const std::string& name, const math::float2& uniform) const
			{
				glUniform2fv(glGetUniformLocation(program, name.c_str()), 1, &uniform.x);
			}

			void Shader::setFloat4x4(const std::string& name, const math::float4x4& uniform) const
			{
				glUniformMatrix4fv(glGetUniformLocation(program, name.c_str()), 1, GL_FALSE, &uniform._m[0]);
			}

			void Shader::setTextureUnit(const std::string& name, unsigned int textureUnit) const
			{
				glUniform1i(glGetUniformLocation(program, name.c_str()), textureUnit);
			}

			void Shader::setFloat(GLuint location, float uniform) const
			{
				glUniform1f(location, uniform);
			}

			void Shader::setFloat4(GLuint location, const math::float4& uniform) const
			{
				glUniform4fv(location, 1, &uniform.x);
			}

			void Shader::setFloat3(GLuint location, const math::float3& uniform) const
			{
				glUniform3fv(location, 1, &uniform.x);
			}

			void Shader::setFloat2(GLuint location, const math::float2& uniform) const
			{
				glUniform2fv(location, 1, &uniform.x);
			}

			void Shader::setFloat4x4(GLuint location, const math::float4x4& uniform) const
			{
				glUniformMatrix4fv(location, 1, GL_FALSE, &uniform._m[0]);
			}

			void Shader::setTextureUnit(GLuint location, unsigned int textureUnit) const
			{
				glUniform1i(location, textureUnit);
			}

			void Shader::bindAttribLocation(const std::string& name, GLuint location)
			{
				glBindAttribLocation(program, location, name.c_str());
			}

			void Shader::bind(unsigned int pass) const
			{
				glUseProgram(program);
			}

			Shader& Shader::operator=(Shader&& other)
			{
				std::swap(vertexShader, other.vertexShader);
				std::swap(fragmentShader, other.fragmentShader);
				std::swap(program, other.program);
				return *this;
			}

			void Shader::setDefaultParameters(PGA::Rendering::GL::Material& material)
			{
			}

		}

	}

}
