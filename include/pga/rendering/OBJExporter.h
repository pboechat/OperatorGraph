#pragma once

#include <string>
#include <ostream>
#include <vector>

#include <math/vector.h>

namespace PGA
{
	namespace Rendering
	{
		struct OBJExporter
		{
			struct Mesh
			{
				std::string name;
				std::string material;
				std::vector<unsigned int> indices;

				Mesh() : name("default_mesh"), material("") {}
				Mesh(const std::string& name, const std::string& material, const std::vector<unsigned int>& indices) : name(name), material(material), indices(indices.begin(), indices.end()) {}

			};

			struct Material
			{
				std::string name;
				math::float3 diffuse;
				math::float3 ambient;
				math::float3 specular;
				float transparency;
				std::string diffuseMap;
				std::string ambientMap;
				std::string specularMap;
				bool useUvX;
				bool useUvY;
				math::float2 uvScale;

				Material() : name("default_material"), diffuse(1.0f, 1.0f, 1.0f), ambient(0.0f, 0.0f, 0.0f), specular(0.0f, 0.0f, 0.0f), transparency(1.0f), diffuseMap(""), ambientMap(""), specularMap(""), useUvX(true), useUvY(true), uvScale(1.0f, 1.0f) {}
				Material(const std::string& name, const math::float3& diffuse, const math::float3& ambient, const math::float3& specular, float transparency, const std::string& diffuseMap, const std::string& ambientMap, const std::string& specularMap)
					: name(name), diffuse(diffuse), ambient(ambient), specular(specular), transparency(transparency), diffuseMap(diffuseMap), ambientMap(ambientMap), specularMap(specularMap) {}

			};

			static void exportMTL(
				std::ostream& out,
				const std::vector<Material>& materials
			);

			static void exportOBJ(
				std::ostream& out,
				const std::vector<math::float4>& vertices,
				const std::vector<math::float2>& uvs,
				const std::vector<math::float3>& normals,
				const std::vector<Mesh>& meshes,
				bool useLeftHandedSystem = true
			);

			static void exportOBJ(
				std::ostream& out,
				const std::vector<math::float4>& vertices, 
				const std::vector<math::float2>& uvs,
				const std::vector<math::float3>& normals, 
				const std::vector<Mesh>& meshes,
				const std::string& mtllib,
				bool useLeftHandedSystem = true
			);

		};

	}

}