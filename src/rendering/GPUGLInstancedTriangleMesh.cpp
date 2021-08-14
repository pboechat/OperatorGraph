#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <pga/core/CUDAException.h>
#include <pga/rendering/GPUGLInstancedTriangleMesh.h>
#include <windows.h>

#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace GL
			{
				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>& source) :
					BaseInstancedTriangleMesh(maxNumInstances, source),
					cudaInstanceAttributesBuffer(0),
					bound(false),
					registered(false)
				{
				}

				InstancedTriangleMesh::InstancedTriangleMesh(size_t maxNumInstances, std::unique_ptr<PGA::Rendering::GL::InstancedTriangleMeshSource>&& source) :
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
							PGA_CUDA_checkedCall(cudaGraphicsGLRegisterBuffer(&cudaInstanceAttributesBuffer, source->getInstanceAttributesBufferRef(), cudaGraphicsMapFlagsReadOnly /* cudaGraphicsMapFlagsWriteDiscard */));
							registered = true;
						}
						PGA_CUDA_checkedCall(cudaGraphicsMapResources(1, &cudaInstanceAttributesBuffer, 0));
						bound = true;
					}
					size_t size;
					PGA_CUDA_checkedCall(cudaGraphicsResourceGetMappedPointer((void**)&data.instancesAttributes, &size, cudaInstanceAttributesBuffer));
				}

				void InstancedTriangleMesh::unbind(const InstancedTriangleMeshData& data)
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
