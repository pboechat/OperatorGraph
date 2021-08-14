#include <pga/rendering/D3DBaseInstancedTriangleMesh.h>

#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			BaseInstancedTriangleMesh::BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>& source) :
				numInstances(0),
				maxNumInstances(maxNumInstances),
				source(std::move(source))
			{
				if (this->maxNumInstances == 0)
					throw std::runtime_error("PGA::Rendering::D3D::BaseInstancedTriangleMesh::ctor(): maximum number of instances must be greater than 0");
				if (this->source == nullptr)
					throw std::runtime_error("PGA::Rendering::D3D::BaseInstancedTriangleMesh::ctor(): source cannot be null");
			}

			BaseInstancedTriangleMesh::BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>&& source) :
				numInstances(0),
				maxNumInstances(maxNumInstances),
				source(std::move(source))
			{
				if (this->maxNumInstances == 0)
					throw std::runtime_error("PGA::Rendering::D3D::BaseInstancedTriangleMesh::ctor(): maximum number of instances must be greater than 0");
				if (this->source == nullptr)
					throw std::runtime_error("PGA::Rendering::D3D::BaseInstancedTriangleMesh::ctor(): source cannot be null");
			}

			size_t BaseInstancedTriangleMesh::getNumInstances() const
			{
				return numInstances;
			}

			size_t BaseInstancedTriangleMesh::getMaxNumInstances() const
			{
				return maxNumInstances;
			}

			void BaseInstancedTriangleMesh::build(ID3D11Device* device, ID3D11DeviceContext* deviceContext)
			{
				source->allocateResources(maxNumInstances, device, deviceContext);
			}

			void BaseInstancedTriangleMesh::draw(ID3D11DeviceContext* deviceContext) const
			{
				if (neverDraw)
					return;
				source->draw(deviceContext);
			}

			size_t BaseInstancedTriangleMesh::appendVertexAttributes(std::vector<math::float4>& positions,
				std::vector<math::float3>& normals,
				std::vector<math::float2>& uvs,
				ID3D11DeviceContext* deviceContext)
			{
				return source->appendVertexAttributes(positions, normals, uvs, deviceContext);
			}

			size_t BaseInstancedTriangleMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset, ID3D11DeviceContext* deviceContext)
			{
				return source->appendIndices(indices, offset, deviceContext);
			}

			bool BaseInstancedTriangleMesh::hasOverflow() const
			{
				return numInstances > maxNumInstances;
			}

		}

	}

}