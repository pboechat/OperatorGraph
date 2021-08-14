#include <pga/core/CUDAException.h>
#include <pga/rendering/GLBaseTriangleMesh.h>
#include <pga/rendering/GLException.h>
#include <pga/rendering/InstancedTriangleMeshData.h>
#include <pga/rendering/RenderingConstants.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			BaseTriangleMesh::BaseTriangleMesh(size_t maxNumVertices, size_t maxNumIndices) :
				numVertices(0),
				numIndices(0),
				vao(0),
				vertexAttributesBuffer(0),
				indexBuffer(0),
				maxNumVertices(maxNumVertices),
				maxNumIndices(maxNumIndices)
			{
				if (this->maxNumVertices == 0)
					throw std::runtime_error("PGA::Rendering::GL::BaseTriangleMesh::ctor(): max. num. vertices for buffer must be > 0");
				if (this->maxNumIndices == 0)
					throw std::runtime_error("PGA::Rendering::GL::BaseTriangleMesh::ctor(): max. num. indices for buffer must be > 0");
			}

			BaseTriangleMesh::~BaseTriangleMesh()
			{
				if (vao)
					glDeleteBuffers(1, &vao);
				if (vertexAttributesBuffer)
					glDeleteBuffers(1, &vertexAttributesBuffer);
				if (indexBuffer)
					glDeleteBuffers(1, &indexBuffer);
				PGA_Rendering_GL_checkError();
			}

			size_t BaseTriangleMesh::getMaxNumVertices() const
			{
				return maxNumVertices;
			}

			size_t BaseTriangleMesh::getMaxNumIndices() const
			{
				return maxNumIndices;
			}

			void BaseTriangleMesh::build()
			{
				assert(sizeof(TriangleMeshData::VertexAttributes) == 48);
				glGenVertexArrays(1, &vao);
				glGenBuffers(1, &vertexAttributesBuffer);
				glGenBuffers(1, &indexBuffer);
				PGA_Rendering_GL_checkError();
				size_t verticesBufferSize = maxNumVertices * sizeof(TriangleMeshData::VertexAttributes);
				size_t indicesBufferSize = maxNumIndices * sizeof(unsigned int);
				glBindVertexArray(vao);
				glBindBuffer(GL_ARRAY_BUFFER, vertexAttributesBuffer);
				glBufferData(GL_ARRAY_BUFFER, verticesBufferSize, 0, GL_DYNAMIC_DRAW);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)0);
				size_t offset = sizeof(math::float4);
				glEnableVertexAttribArray(1);
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)offset);
				offset += sizeof(math::float3);
				glEnableVertexAttribArray(2);
				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(TriangleMeshData::VertexAttributes), (void*)offset);
				// NOTE: the model matrix vertex attribute has to be initialized with a identity matrix
				// so that shaders can work transparently with both triangle mesh and instanced triangle mesh
				assert(sizeof(InstancedTriangleMeshData::InstanceAttributes) / sizeof(math::float4) == 5);
				glVertexAttrib4f(PGA::Rendering::Constants::InstanceAttributesLocation, 1.0f, 0.0f, 0.0f, 0.0f);
				glVertexAttrib4f(PGA::Rendering::Constants::InstanceAttributesLocation + 1, 0.0f, 1.0f, 0.0f, 0.0f);
				glVertexAttrib4f(PGA::Rendering::Constants::InstanceAttributesLocation + 2, 0.0f, 0.0f, 1.0f, 0.0f);
				glVertexAttrib4f(PGA::Rendering::Constants::InstanceAttributesLocation + 3, 0.0f, 0.0f, 0.0f, 1.0f);
				glVertexAttrib4f(PGA::Rendering::Constants::InstanceAttributesLocation + 4, 0.0f, 0.0f, 0.0f, 0.0f);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesBufferSize, 0, GL_DYNAMIC_DRAW);
				glBindVertexArray(0);
				PGA_Rendering_GL_checkError();
			}

			void BaseTriangleMesh::draw() const
			{
				if (neverDraw)
					return;
				if (numIndices == 0)
					return;
				glBindVertexArray(vao);
				glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(numIndices), GL_UNSIGNED_INT, 0);
			}

			bool BaseTriangleMesh::hasOverflown() const
			{
				return numVertices > maxNumVertices || numIndices > maxNumIndices;
			}

			size_t BaseTriangleMesh::getNumVertices() const
			{
				return numVertices;
			}

			size_t BaseTriangleMesh::getNumIndices() const
			{
				return numIndices;
			}

		}

	}

}
