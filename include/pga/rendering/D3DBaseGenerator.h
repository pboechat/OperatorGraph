#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <d3d11.h>

#include <math/vector.h>

#include "D3DMesh.h"
#include "D3DMaterial.h"
#include "OBJExporter.h"

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class BaseGenerator
			{
			protected:
				void exportMeshToOBJ
					(
						PGA::Rendering::D3D::Mesh* mesh,
						std::vector<math::float4>& positions,
						std::vector<math::float3>& normals,
						std::vector<math::float2>& uvs,
						std::vector<PGA::Rendering::OBJExporter::Mesh>& objMeshes,
						size_t& offset,
						const std::string& baseFileName,
						const std::string& meshName,
						const std::string& materialName,
						bool useUvX,
						bool useUvY,
						const math::float2& uvScale,
						ID3D11DeviceContext* deviceContext
					)
				{
					std::vector<unsigned int> indices;
					size_t c = mesh->appendVertexAttributes(positions, normals, uvs, deviceContext);
					if (c == 0)
						return;
					mesh->appendIndices(indices, offset, deviceContext);
					scaleUVs(useUvX, useUvY, uvScale, indices, positions, normals, uvs, offset);
					objMeshes.emplace_back(meshName, materialName, indices);
					offset += c;
				}

				void scaleUVs
					(
						bool useUvX,
						bool useUvY,
						const math::float2& scale,
						std::vector<unsigned int>& indices,
						std::vector<math::float4>& positions,
						std::vector<math::float3>& normals,
						std::vector<math::float2>&uvs,
						size_t offset
					)
				{
					if (useUvX && useUvY)
					{
						auto it = uvs.begin();
						std::advance(it, offset);
						while (it != uvs.end())
						{
							auto& uv = *it;
							uv.x *= scale.x;
							uv.y *= scale.y;
							it++;
						}
					}
					else
					{
						for (auto i = 0; i < indices.size(); i++)
						{
							auto index = indices[i];
							const auto& position = positions[index];
							auto n = abs(normals[index]);
							auto& uv = uvs[index];
							if (n.x > n.y)
							{
								if (n.x > n.z)
									uv = math::float2((useUvX) ? uv.x : position.z, (useUvY) ? uv.y : position.y);
								else
									uv = math::float2((useUvX) ? uv.x : position.x, (useUvY) ? uv.y : position.y);
							}
							else
							{
								if (n.y > n.z)
									uv = math::float2((useUvX) ? uv.x : position.x, (useUvY) ? uv.y : position.z);
								else
									uv = math::float2((useUvX) ? uv.x : position.x, (useUvY) ? uv.y : position.y);
							}
							uv.x *= scale.x;
							uv.y *= scale.y;
						}
					}
				}

				BaseGenerator() = default;

			public:
				void exportToOBJ
					(
						std::ostream& objOut,
						const std::string& name,
						bool useLeftHandedCoordsSystem,
						ID3D11DeviceContext* deviceContext
					)
				{
					std::map<unsigned int, PGA::Rendering::OBJExporter::Material> materials;
					std::stringstream mtlOut;
					exportToOBJ(objOut, mtlOut, name, "", materials, useLeftHandedCoordsSystem, deviceContext);
				}

				void exportToOBJ
					(
						std::ostream& objOut,
						std::ostream& mtlOut,
						const std::string& name,
						const std::string& mtlFileName,
						std::map<unsigned int, PGA::Rendering::OBJExporter::Material>& materials,
						bool useLeftHandedCoordsSystem,
						ID3D11DeviceContext* deviceContext
					)
				{
					std::vector<math::float4> positions(0);
					std::vector<math::float3> normals(0);
					std::vector<math::float2> uvs(0);
					std::vector<PGA::Rendering::OBJExporter::Material> objMaterials;
					std::vector<PGA::Rendering::OBJExporter::Mesh> objMeshes;
					size_t offset = 0;
					for (auto i = 0; i < getNumMeshes(); i++)
					{
						std::string meshName = "mesh_" + std::to_string(i);
						std::string materialName;
						bool useUvX = true, useUvY = true;
						math::float2 uvScale(1.0f, 1.0f);
						auto it = materials.find(i);
						if (it != materials.end())
						{
							materialName = it->second.name;
							useUvX = it->second.useUvX;
							useUvY = it->second.useUvY;
							uvScale = it->second.uvScale;
							objMaterials.emplace_back(it->second);
						}
						exportMeshToOBJ(
							getMeshPtr(i),
							positions,
							normals,
							uvs,
							objMeshes,
							offset,
							name,
							meshName,
							materialName,
							useUvX,
							useUvY,
							uvScale,
							deviceContext
						);
					}
					PGA::Rendering::OBJExporter::exportOBJ(
						objOut,
						positions,
						uvs,
						normals,
						objMeshes,
						mtlFileName,
						useLeftHandedCoordsSystem
					);
					PGA::Rendering::OBJExporter::exportMTL(mtlOut, objMaterials);
				}

				virtual size_t getNumMeshes() const = 0;
				virtual PGA::Rendering::D3D::Mesh* getMeshPtr(size_t i) const = 0;

			};

		}

	}

}