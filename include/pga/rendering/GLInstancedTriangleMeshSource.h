#pragma once

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>

#include "GLMesh.h"
#include "InstancedTriangleMeshData.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			class InstancedTriangleMeshSource : public PGA::Rendering::GL::Mesh
			{
			public:
				InstancedTriangleMeshSource() = default;

				virtual size_t getNumInstances() = 0;
				virtual void sync(size_t numInstances) = 0;
				virtual void sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes) = 0;
				virtual void allocateResources(size_t maxNumInstances) = 0;
				virtual GLuint getInstanceAttributesBufferRef() = 0;

			};

		}

	}

}
