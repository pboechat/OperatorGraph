#pragma once

#include <vector>
#include <memory>
#include <d3d11.h>

#include <math/vector.h>

#include "D3DMesh.h"
#include "D3DInstancedTriangleMeshSource.h"
#include "InstancedTriangleMeshData.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class BaseInstancedTriangleMesh : public PGA::Rendering::D3D::Mesh
			{
			protected:
				size_t numInstances;
				size_t maxNumInstances;
				std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource> source;

			public:
				BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>& source);
				BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>&& source);

				bool hasOverflow() const;
				size_t getNumInstances() const;
				size_t getMaxNumInstances() const;
				void build(ID3D11Device* device, ID3D11DeviceContext* deviceContext);
				virtual void bind(InstancedTriangleMeshData& data) = 0;
				virtual void unbind(const InstancedTriangleMeshData& data, ID3D11DeviceContext* deviceContext) = 0;
				virtual void draw(ID3D11DeviceContext* deviceContext) const;
				virtual size_t appendVertexAttributes(std::vector<math::float4>& positions,
					std::vector<math::float3>& normals,
					std::vector<math::float2>& uvs,
					ID3D11DeviceContext* deviceContext);
				virtual size_t appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext);

			};

		}

	}

}

