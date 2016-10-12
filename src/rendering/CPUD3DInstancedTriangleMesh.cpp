#include <pga/rendering/CPUD3DInstancedTriangleMesh.h>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace D3D
			{
				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>& source) :
					BaseInstancedTriangleMesh(maxNumInstances, source)
				{
				}

				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumElements, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>&& source) :
					BaseInstancedTriangleMesh(maxNumElements, std::move(source))
				{
				}

				void InstancedTriangleMesh::bind(InstancedTriangleMeshData& data)
				{
					data.numInstances = 0;
					data.maxNumInstances = static_cast<unsigned int>(maxNumInstances);
					data.instancesAttributes = (InstancedTriangleMeshData::InstanceAttributes*)malloc(maxNumInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes));
				}

				void InstancedTriangleMesh::unbind(const InstancedTriangleMeshData& data, ID3D11DeviceContext* deviceContext)
				{
					numInstances = data.numInstances;
					source->sync(((numInstances > maxNumInstances) ? maxNumInstances : numInstances), data.instancesAttributes, deviceContext);
					free(data.instancesAttributes);
				}

			}

		}

	}

}
