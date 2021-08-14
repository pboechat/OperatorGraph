#pragma once

#include "D3DBaseGenerator.h"
#include "D3DMesh.h"
#include "GPUD3DInstancedTriangleMesh.h"
#include "GPUD3DTriangleMesh.h"
#include "InstancedTriangleMeshData.h"
#include "RenderingGlobalVariables.cuh"
#include "TriangleMeshData.h"

#include <d3d11.h>

#include <memory>
#include <string>
#include <vector>

// NOTE: cannot be moved to a CPP file because of RenderingGlobalVariables.cuh
namespace PGA
{
	namespace Rendering
	{
		namespace GPU
		{
			namespace D3D
			{
				class Generator : public PGA::Rendering::D3D::BaseGenerator
				{
				protected:
					std::vector<std::unique_ptr<PGA::Rendering::GPU::D3D::TriangleMesh>> triangleMeshes;
					std::vector<std::unique_ptr<PGA::Rendering::GPU::D3D::InstancedTriangleMesh>> instancedTriangleMeshes;

					virtual size_t getNumMeshes() const
					{
						return triangleMeshes.size() + instancedTriangleMeshes.size();
					}

					virtual PGA::Rendering::D3D::Mesh* getMeshPtr(size_t i) const
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

					void build(std::vector<std::unique_ptr<PGA::Rendering::GPU::D3D::TriangleMesh>>&& triangleMeshes, 
						std::vector<std::unique_ptr<PGA::Rendering::GPU::D3D::InstancedTriangleMesh>>&& instancedTriangleMeshes,
						ID3D11Device* device,
						ID3D11DeviceContext* deviceContext)
					{
						this->triangleMeshes.clear();
						for (auto& triangleMesh : triangleMeshes)
						{
							triangleMesh->build(device, deviceContext);
							this->triangleMeshes.emplace_back(std::move(triangleMesh));
						}
						this->instancedTriangleMeshes.clear();
						for (auto& instancedTriangleMesh : instancedTriangleMeshes)
						{
							instancedTriangleMesh->build(device, deviceContext);
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

					void unbind(ID3D11DeviceContext* deviceContext)
					{
						if (!triangleMeshes.empty())
						{
							std::unique_ptr<TriangleMeshData[]> triangleMeshesData(new TriangleMeshData[triangleMeshes.size()]);
							Device::getTriangleMeshes(static_cast<unsigned int>(triangleMeshes.size()), triangleMeshesData);
							for (auto i = 0; i < triangleMeshes.size(); i++)
								triangleMeshes[i]->unbind(triangleMeshesData[i], deviceContext);
						}
						if (!instancedTriangleMeshes.empty())
						{
							std::unique_ptr<InstancedTriangleMeshData[]> instancedTriangleMeshesData(new InstancedTriangleMeshData[instancedTriangleMeshes.size()]);
							Device::getInstancedTriangleMeshes(static_cast<unsigned int>(instancedTriangleMeshes.size()), instancedTriangleMeshesData);
							for (auto i = 0; i < instancedTriangleMeshes.size(); i++)
								instancedTriangleMeshes[i]->unbind(instancedTriangleMeshesData[i], deviceContext);
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

					std::unique_ptr<PGA::Rendering::GPU::D3D::TriangleMesh>& getTriangleMesh(size_t i)
					{
						return triangleMeshes[i];
					}

					const std::unique_ptr<PGA::Rendering::GPU::D3D::TriangleMesh>& getTriangleMesh(size_t i) const
					{
						return triangleMeshes[i];
					}

					std::unique_ptr<PGA::Rendering::GPU::D3D::InstancedTriangleMesh>& getInstancedTriangleMesh(size_t i)
					{
						return instancedTriangleMeshes[i];
					}

					const std::unique_ptr<PGA::Rendering::GPU::D3D::InstancedTriangleMesh>& getInstancedTriangleMesh(size_t i) const
					{
						return instancedTriangleMeshes[i];
					}

				};

			}

		}

	}

}
