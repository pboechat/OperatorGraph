#pragma once

#include "GLBaseInstancedTriangleMesh.h"
#include "GLInstancedTriangleMeshSource.h"

#include <driver_types.h>

#include <memory>

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace GL
			{
				class InstancedTriangleMesh : public PGA::Rendering::GL::BaseInstancedTriangleMesh
				{
				private:
					cudaGraphicsResource_t cudaInstanceAttributesBuffer;
					bool bound;
					bool registered;

				public:
					InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>& source);
					InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>&& source);
					virtual ~InstancedTriangleMesh();

					virtual void bind(InstancedTriangleMeshData& data);
					virtual void unbind(const InstancedTriangleMeshData& data);

				};

			}

		}

	}

}
