#pragma once

#include "D3DInstancedTriangleMeshSource.h"
#include "tiny_obj_loader.h"

#include <d3d11.h>
#include <math/matrix.h>
#include <math/vector.h>

#include <memory>
#include <string>
#include <vector>

// NOTE: https://github.com/syoyo/tinyobjloader
namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class OBJMesh : public PGA::Rendering::D3D::InstancedTriangleMeshSource
			{
			private:
				class OBJShape
				{
				private:
					const tinyobj::shape_t& shape;
					ID3D11Buffer* vertexAttributesBuffer;
					ID3D11Buffer* indexBuffer;

				public:
					OBJShape(const tinyobj::shape_t& objShape);
					~OBJShape();
					void allocateResources(ID3D11Buffer* instanceAttributesBuffer, const std::vector<tinyobj::material_t>& objMaterials, ID3D11Device* device, ID3D11DeviceContext* deviceContext);
					void draw(size_t numInstances, ID3D11Buffer* instanceAttributesBuffer, ID3D11DeviceContext* deviceContext) const;
					size_t appendVertexAttributes(std::vector<math::float4>& positions,
						std::vector<math::float3>& normals,
						std::vector<math::float2>& uvs,
						const std::vector<InstancedTriangleMeshData::InstanceAttributes>& instancesAttributes) const;
					size_t appendIndices(std::vector<unsigned int>& indices, size_t offset, size_t numInstances) const;

				};

				ID3D11Buffer* instanceAttributesBuffer;
				std::vector<InstancedTriangleMeshData::InstanceAttributes> instancesAttributes;
				std::vector<tinyobj::shape_t> objShapes;
				std::vector<tinyobj::material_t> objMaterials;
				std::vector<std::unique_ptr<OBJShape>> subMeshes;
				bool pendingSync;

				void syncInstanceAttributes(ID3D11DeviceContext* deviceContext);

			public:
				OBJMesh(const std::string& fileName);
				OBJMesh(OBJMesh&) = delete;
				virtual ~OBJMesh();
				OBJMesh& operator=(OBJMesh&) = delete;

				virtual size_t getNumInstances();
				virtual void sync(size_t numInstances);
				virtual void sync(size_t numInstances, const InstancedTriangleMeshData::InstanceAttributes* instancesAttributes, ID3D11DeviceContext* deviceContext);
				virtual void allocateResources(size_t maxNumInstances, ID3D11Device* device, ID3D11DeviceContext* deviceContext);
				virtual ID3D11Buffer* getInstanceAttributesBufferRef();
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
