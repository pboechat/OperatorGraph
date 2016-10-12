#pragma once

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>

#include <math/vector.h>

#include "GLMesh.h"
#include "TriangleMeshData.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class BaseTriangleMesh : public PGA::Rendering::GL::Mesh
			{
			protected:
				GLuint vao;
				GLuint vertexAttributesBuffer;
				GLuint indexBuffer;
				size_t numVertices;
				size_t numIndices;
				size_t maxNumVertices;
				size_t maxNumIndices;

				BaseTriangleMesh(size_t maxNumVertices, size_t maxNumIndices);

			public:
				virtual ~BaseTriangleMesh();

				bool hasOverflown() const;
				size_t getNumVertices() const;
				size_t getNumIndices() const;
				size_t getMaxNumVertices() const;
				size_t getMaxNumIndices() const;
				virtual void draw() const;
				void build();
				virtual void bind(TriangleMeshData& data) = 0;
				virtual void unbind(const TriangleMeshData& data) = 0;

			};

		}

	}

}