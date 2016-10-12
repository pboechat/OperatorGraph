#pragma once

#include <vector>
#include <memory>

#include <math/vector.h>

#include "GLMesh.h"
#include "GLInstancedTriangleMeshSource.h"
#include "InstancedTriangleMeshData.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class BaseInstancedTriangleMesh : public PGA::Rendering::GL::Mesh
			{
			protected:
				size_t numInstances;
				size_t maxNumInstances;
				std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource> source;

			public:
				BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>& source);
				BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>&& source);

				bool hasOverflow() const;
				size_t getNumInstances() const;
				size_t getMaxNumInstances() const;
				void build();
				virtual void bind(InstancedTriangleMeshData& data) = 0;
				virtual void unbind(const InstancedTriangleMeshData& data) = 0;
				virtual void draw() const;
				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs);
				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset /* = 0 */);

			};

		}

	}

}
