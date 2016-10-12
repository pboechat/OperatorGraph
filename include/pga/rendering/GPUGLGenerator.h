#pragma once

#include <string>
#include <vector>
#include <memory>

// NOTE: cannot be moved to a CPP file because of RenderingGlobalVariables.cuh
#include "RenderingGlobalVariables.cuh"
#include "GLMesh.h"
#include "GPUGLTriangleMesh.h"
#include "GPUGLInstancedTriangleMesh.h"
#include "TriangleMeshData.h"
#include "InstancedTriangleMeshData.h"
#include "GLBaseGenerator.h"

namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace GL
			{
				class Generator : public PGA::Rendering::GL::BaseGenerator
				{
				protected:
					std::vector<std::unique_ptr<PGA::Rendering::GPU::GL::TriangleMesh>> triangleMeshes;
					std::vector<std::unique_ptr<PGA::Rendering::GPU::GL::InstancedTriangleMesh>> instancedTriangleMeshes;

					virtual size_t getNumMeshes() const
					{
						return triangleMeshes.size() + instancedTriangleMeshes.size();
					}

					virtual PGA::Rendering::GL::Mesh* getMeshPtr(size_t i) const
					{
						if (i < triangleMeshes.size())
							return triangleMeshes[i].get();
						else
							return instancedTriangleMeshes[i - triangleMeshes.size()].get();
					}

				public:
					Generator() = default;

					~Generator()
					{
						triangleMeshes.clear();
						instancedTriangleMeshes.clear();
					}

					size_t getTotalNumVertices() const
					{
						size_t c = 0;
						for (auto& triangleMesh : triangleMeshes)
							c += triangleMesh->getNumVertices();
						return c;
					}

					size_t getTotalNumIndices() const
					{
						size_t c = 0;
						for (auto& triangleMesh : triangleMeshes)
							c += triangleMesh->getNumIndices();
						return c;
					}

					size_t getTotalNumInstances() const
					{
						size_t c = 0;
						for (auto& instancedTriangleMesh : instancedTriangleMeshes)
							c += instancedTriangleMesh->getNumInstances();
						return c;
					}

					void build(std::vector<std::unique_ptr<PGA::Rendering::GPU::GL::TriangleMesh>>&& triangleMeshes, 
						std::vector<std::unique_ptr<PGA::Rendering::GPU::GL::InstancedTriangleMesh>>&& instancedTriangleMeshes)
					{
						this->triangleMeshes.clear();
						for (auto& triangleMesh : triangleMeshes)
						{
							triangleMesh->build();
							this->triangleMeshes.emplace_back(std::move(triangleMesh));
						}
						this->instancedTriangleMeshes.clear();
						for (auto& instancedTriangleMesh : instancedTriangleMeshes)
						{
							instancedTriangleMesh->build();
							this->instancedTriangleMeshes.emplace_back(std::move(instancedTriangleMesh));
						}
					}

					void bind()
					{
						unsigned int numTriangleMeshes = static_cast<unsigned int>(this->triangleMeshes.size());
						Device::setNumTriangleMeshes(numTriangleMeshes);
						if (!this->triangleMeshes.empty())
						{
							std::unique_ptr<TriangleMeshData[]> triangleMeshesData(new TriangleMeshData[numTriangleMeshes]);
							for (auto i = 0; i < this->triangleMeshes.size(); i++)
								this->triangleMeshes[i]->bind(triangleMeshesData[i]);
							Device::setTriangleMeshes(numTriangleMeshes, triangleMeshesData);
						}
						unsigned int numInstancedTriangleMeshes = static_cast<unsigned int>(this->instancedTriangleMeshes.size());
						Device::setNumInstancedTriangleMeshes(numInstancedTriangleMeshes);
						if (!this->instancedTriangleMeshes.empty())
						{
							std::unique_ptr<InstancedTriangleMeshData[]> instancedTriangleMeshesData(new InstancedTriangleMeshData[numInstancedTriangleMeshes]);
							for (auto i = 0; i < this->instancedTriangleMeshes.size(); i++)
								this->instancedTriangleMeshes[i]->bind(instancedTriangleMeshesData[i]);
							Device::setInstancedTriangleMeshes(numInstancedTriangleMeshes, instancedTriangleMeshesData);
						}
					}

					void unbind()
					{
						if (!triangleMeshes.empty())
						{
							std::unique_ptr<TriangleMeshData[]> triangleMeshesData(new TriangleMeshData[triangleMeshes.size()]);
							Device::getTriangleMeshes(static_cast<unsigned int>(triangleMeshes.size()), triangleMeshesData);
							for (auto i = 0; i < triangleMeshes.size(); i++)
								triangleMeshes[i]->unbind(triangleMeshesData[i]);
						}
						if (!instancedTriangleMeshes.empty())
						{
							std::unique_ptr<InstancedTriangleMeshData[]> instancedTriangleMeshesData(new InstancedTriangleMeshData[instancedTriangleMeshes.size()]);
							Device::getInstancedTriangleMeshes(static_cast<unsigned int>(instancedTriangleMeshes.size()), instancedTriangleMeshesData);
							for (auto i = 0; i < instancedTriangleMeshes.size(); i++)
								instancedTriangleMeshes[i]->unbind(instancedTriangleMeshesData[i]);
						}
					}

					size_t getNumTriangleMeshes() const
					{
						return triangleMeshes.size();
					}

					size_t getNumInstancedTriangleMeshes() const
					{
						return instancedTriangleMeshes.size();
					}

					std::unique_ptr<PGA::Rendering::GPU::GL::TriangleMesh>& getTriangleMesh(size_t i)
					{
						return triangleMeshes[i];
					}

					const std::unique_ptr<PGA::Rendering::GPU::GL::TriangleMesh>& getTriangleMesh(size_t i) const
					{
						return triangleMeshes[i];
					}

					std::unique_ptr<PGA::Rendering::GPU::GL::InstancedTriangleMesh>& getInstancedTriangleMesh(size_t i)
					{
						return instancedTriangleMeshes[i];
					}

					const std::unique_ptr<PGA::Rendering::GPU::GL::InstancedTriangleMesh>& getInstancedTriangleMesh(size_t i) const
					{
						return instancedTriangleMeshes[i];
					}

				};

			}

		}

	}

}
