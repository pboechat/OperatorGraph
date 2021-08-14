#pragma once

#include "D3DBaseInstancedTriangleMesh.h"
#include "D3DInstancedTriangleMeshSource.h"

#include <memory>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace D3D
			{
				class InstancedTriangleMesh : public PGA::Rendering::D3D::BaseInstancedTriangleMesh
				{
				public:
					InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>& source);
					InstancedTriangleMesh(size_t maxNumElements, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>&& source);

					virtual void bind(InstancedTriangleMeshData& data);
					virtual void unbind(const InstancedTriangleMeshData& data, ID3D11DeviceContext* deviceContext);

				};

			}

		}

	}

}