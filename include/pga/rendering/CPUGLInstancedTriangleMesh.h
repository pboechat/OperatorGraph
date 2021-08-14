#pragma once

#include "GLBaseInstancedTriangleMesh.h"
#include "GLInstancedTriangleMeshSource.h"

#include <memory>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace GL
			{
				class InstancedTriangleMesh : public PGA::Rendering::GL::BaseInstancedTriangleMesh
				{
				public:
					InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>& source);
					InstancedTriangleMesh(size_t maxNumElements, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>&& source);

					virtual void bind(InstancedTriangleMeshData& data);
					virtual void unbind(const InstancedTriangleMeshData& data);

				};

			}

		}

	}

}