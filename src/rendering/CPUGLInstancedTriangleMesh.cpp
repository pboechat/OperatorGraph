#include <pga/rendering/CPUGLInstancedTriangleMesh.h>

namespace PGA
{
	namespace Rendering
	{
		namespace CPU
		{
			namespace GL
			{
				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>& source) :
					BaseInstancedTriangleMesh(maxNumInstances, source)
				{
				}

				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumElements, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>&& source) :
					BaseInstancedTriangleMesh(maxNumElements, std::move(source))
				{
				}

				void InstancedTriangleMesh::bind(InstancedTriangleMeshData& data)
				{
					data.numInstances = 0;
					data.maxNumInstances = static_cast<unsigned int>(maxNumInstances);
					data.instancesAttributes = (InstancedTriangleMeshData::InstanceAttributes*)malloc(maxNumInstances * sizeof(InstancedTriangleMeshData::InstanceAttributes));
				}

				void InstancedTriangleMesh::unbind(const InstancedTriangleMeshData& data)
				{
					numInstances = data.numInstances;
					source->sync(((numInstances > maxNumInstances) ? maxNumInstances : numInstances), data.instancesAttributes);
					free(data.instancesAttributes);
				}
			}

		}

	}

}