#pragma once

#include <d3d11.h>

#include "D3DMesh.h"
#include "InstancedTriangleMeshData.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class InstancedTriangleMeshSource : public PGA::Rendering::D3D::Mesh
			{
			public:
				InstancedTriangleMeshSource() = default;

				virtual size_t getNumInstances() = 0;
				virtual void sync(size_t numInstances) = 0;
				virtual void sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes, ID3D11DeviceContext* deviceContext) = 0;
				virtual void allocateResources(size_t maxNumInstances, ID3D11Device* device, ID3D11DeviceContext* deviceContext) = 0;
				virtual ID3D11Buffer* getInstanceAttributesBufferRef() = 0;

			};

		}

	}

}
