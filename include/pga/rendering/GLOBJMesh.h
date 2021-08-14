#pragma once

#include "GLInstancedTriangleMeshSource.h"
#include "tiny_obj_loader.h"

#include <GL/glew.h>
#include <math/matrix.h>
#include <math/vector.h>
#include <windows.h>

#include <memory>
#include <string>
#include <vector>

// NOTE: https://github.com/syoyo/tinyobjloader
namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class OBJMesh : public PGA::Rendering::GL::InstancedTriangleMeshSource
			{
			private:
				class OBJShape
				{
				private:
					const tinyobj::shape_t& shape;
					GLuint vao;
					GLuint vertexAttributesBuffer;
					GLuint indexBuffer;

				public:
					OBJShape(const tinyobj::shape_t& objShape);
					~OBJShape();
					void allocateResources(GLuint instanceAttributesBuffer, const std::vector<tinyobj::material_t>& objMaterials);
					void draw(size_t numInstances) const;
					size_t appendVertexAttributes(std::vector<math::float4>& positions,
						std::vector<math::float3>& normals,
						std::vector<math::float2>& uvs,
						const std::vector<InstancedTriangleMeshData::InstanceAttributes>& instancesAttributes) const;
					size_t appendIndices(std::vector<unsigned int>& indices, size_t offset, size_t numInstances) const;

				};

				GLuint instanceAttributesBuffer;
				std::vector<InstancedTriangleMeshData::InstanceAttributes> instancesAttributes;
				std::vector<tinyobj::shape_t> objShapes;
				std::vector<tinyobj::material_t> objMaterials;
				std::vector<std::unique_ptr<OBJShape>> subMeshes;
				bool pendingSync;

			public:
				OBJMesh(const std::string& fileName);
				~OBJMesh();

				virtual size_t getNumInstances();
				virtual void sync(size_t numInstances);
				virtual void sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes);
				virtual void allocateResources(size_t maxNumInstances);
				virtual GLuint getInstanceAttributesBufferRef();
				virtual void draw() const;
				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs);
				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset /* = 0 */);

			};

		}
	}

}
