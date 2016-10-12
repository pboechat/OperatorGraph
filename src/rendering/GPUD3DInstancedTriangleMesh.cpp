#include <stdexcept>
#include <cuda_d3d11_interop.h>

#include <pga/core/CUDAException.h>
#include <pga/rendering/GPUD3DInstancedTriangleMesh.h>

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace D3D
			{
				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>& source) :
					BaseInstancedTriangleMesh(maxNumInstances, source),
					cudaInstanceAttributesBuffer(0),
					bound(false),
					registered(false)
				{
				}

				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::D3D::InstancedTriangleMeshSource>&& source) :
					BaseInstancedTriangleMesh(maxNumInstances, source),
					cudaInstanceAttributesBuffer(0),
					bound(false),
					registered(false)
				{
				}

				InstancedTriangleMesh::~InstancedTriangleMesh()
				{
					if (registered)
					{
						if (bound)
							PGA_CUDA_checkedCall(cudaGraphicsUnmapResources(1, &cudaInstanceAttributesBuffer, 0));
						PGA_CUDA_checkedCall(cudaGraphicsUnregisterResource(cudaInstanceAttributesBuffer));
					}
				}

				void InstancedTriangleMesh::bind(InstancedTriangleMeshData& data)
				{
					data.numInstances = 0;
					data.maxNumInstances = static_cast<unsigned int>(maxNumInstances);
					if (!bound)
					{
						if (!registered)
						{
							PGA_CUDA_checkedCall(cudaGraphicsD3D11RegisterResource(&cudaInstanceAttributesBuffer, source->getInstanceAttributesBufferRef(), cudaGraphicsRegisterFlagsNone /* the only flag working for now */));
							registered = true;
						}
						PGA_CUDA_checkedCall(cudaGraphicsMapResources(1, &cudaInstanceAttributesBuffer, 0));
						bound = true;
					}
					size_t size;
					PGA_CUDA_checkedCall(cudaGraphicsResourceGetMappedPointer((void**)&data.instancesAttributes, &size, cudaInstanceAttributesBuffer));
				}

				void InstancedTriangleMesh::unbind(const InstancedTriangleMeshData& data, ID3D11DeviceContext* deviceContext)
				{
					if (neverDraw)
						return;
					if (!bound)
						return;
					numInstances = data.numInstances;
					source->sync((numInstances > maxNumInstances) ? maxNumInstances : numInstances);
					PGA_CUDA_checkedCall(cudaGraphicsUnmapResources(1, &cudaInstanceAttributesBuffer, 0));
					bound = false;
				}

			}

		}

	}

}
