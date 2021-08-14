#include <pga/rendering/GLBaseInstancedTriangleMesh.h>

#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace GL
		{
			BaseInstancedTriangleMesh::BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>& source) :
				numInstances(0),
				maxNumInstances(maxNumInstances),
				source(std::move(source))
			{
				if (this->maxNumInstances == 0)
					throw std::runtime_error("PGA::Rendering::GL::BaseInstancedTriangleMesh::ctor(): maximum number of instances must be greater than 0");
				if (this->source == nullptr)
					throw std::runtime_error("PGA::Rendering::GL::BaseInstancedTriangleMesh::ctor(): source cannot be null");
			}

			BaseInstancedTriangleMesh::BaseInstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>&& source) :
				numInstances(0),
				maxNumInstances(maxNumInstances),
				source(std::move(source))
			{
				if (this->maxNumInstances == 0)
					throw std::runtime_error("PGA::Rendering::GL::BaseInstancedTriangleMesh::ctor(): maximum number of instances must be greater than 0");
				if (this->source == nullptr)
					throw std::runtime_error("PGA::Rendering::GL::BaseInstancedTriangleMesh::ctor(): source cannot be null");
			}

			size_t BaseInstancedTriangleMesh::getNumInstances() const
			{
				return numInstances;
			}

			size_t BaseInstancedTriangleMesh::getMaxNumInstances() const
			{
				return maxNumInstances;
			}

			void BaseInstancedTriangleMesh::build()
			{
				source->allocateResources(maxNumInstances);
			}

			void BaseInstancedTriangleMesh::draw() const
			{
				if (neverDraw)
					return;
				source->draw();
			}

			size_t BaseInstancedTriangleMesh::appendVertexAttributes(std::vector<math::float4>& positions,
				std::vector<math::float3>& normals,
				std::vector<math::float2>& uvs)
			{
				return source->appendVertexAttributes(positions, normals, uvs);
			}

			size_t BaseInstancedTriangleMesh::appendIndices(std::vector<unsigned int>& indices, size_t offset)
			{
				return source->appendIndices(indices, offset);
			}

			bool BaseInstancedTriangleMesh::hasOverflow() const
			{
				return numInstances > maxNumInstances;
			}

		}

	}

}