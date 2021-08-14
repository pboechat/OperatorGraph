#pragma once

#include "D3DBaseInstancedTriangleMesh.h"
#include "D3DInstancedTriangleMeshSource.h"

#include <driver_types.h>

#include <memory>

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace D3D
			{
				class InstancedTriangleMesh : public PGA::Rendering::D3D::BaseInstancedTriangleMesh
				{
				private:
					cudaGraphicsResource_t cudaInstanceAttributesBuffer;
					bool bound;
					bool registered;

				public:
					InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>& source);
					InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>&& source);
					virtual ~InstancedTriangleMesh();

					virtual void bind(InstancedTriangleMeshData& data);
					virtual void unbind(const InstancedTriangleMeshData& data, ID3D11DeviceContext* deviceContext);

				};

			}

		}

	}

}
