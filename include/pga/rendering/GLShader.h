#pragma once

#include "GLMaterial.h"

#include <GL/glew.h>
#include <math/matrix.h>
#include <math/vector.h>
#include <windows.h>

#include <initializer_list>
#include <string>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class Shader
			{
			private:
				Shader(const Shader&) = delete;
				Shader& operator =(Shader&) = delete;

			protected:
				GLuint program;
				GLuint vertexShader;
				GLuint geometryShader;
				GLuint fragmentShader;

				virtual void setDefaultParameters(PGA::Rendering::GL::Material& material);

				friend PGA::Rendering::GL::Material;

			public:
				static void setIncludePath(const std::initializer_list<std::string>& includePath);

				Shader() = delete;
				virtual ~Shader();
				Shader(const std::string& vertexShaderFileName, const std::string& fragmentShaderFileName);
				Shader(const std::string& vertexShaderFileName, const std::string& geometryShaderFileName, const std::string& fragmentShaderFileName);
				Shader(Shader&& other);

				Shader& operator=(Shader&& other);

				virtual void bind(unsigned int pass) const;
				void bindAttribLocation(const std::string& name, GLuint index);
				GLuint getUniformLocation(const std::string& name) const;
				void setFloat4x4(const std::string& name, const math::float4x4& uniform) const;
				void setFloat2(const std::string& name, const math::float2& uniform) const;
				void setFloat3(const std::string& name, const math::float3& uniform) const;
				void setFloat4(const std::string& name, const math::float4& uniform) const;
				void setFloat(const std::string& name, float uniform) const;
				void setTextureUnit(const std::string& name, unsigned int textureUnit) const;
				void setFloat4x4(GLuint location, const math::float4x4& uniform) const;
				void setFloat2(GLuint location, const math::float2& uniform) const;
				void setFloat3(GLuint location, const math::float3& uniform) const;
				void setFloat4(GLuint location, const math::float4& uniform) const;
				void setFloat(GLuint location, float uniform) const;
				void setTextureUnit(GLuint location, unsigned int textureUnit) const;

			};

		}

	}

}
