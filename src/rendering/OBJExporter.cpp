#include <pga/rendering/OBJExporter.h>

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace PGA
{
	namespace Rendering
	{
		void OBJExporter::exportMTL(std::ostream& out, const std::vector<Material>& materials)
		{
			for (const auto& material : materials)
			{
				out << "newmtl " << material.name << std::endl;
				out << "Kd " << material.diffuse.x << " " << material.diffuse.y << " " << material.diffuse.z << std::endl;
				out << "Ka " << material.ambient.x << " " << material.ambient.y << " " << material.ambient.z << std::endl;
				out << "Ks " << material.specular.x << " " << material.specular.y << " " << material.specular.z << std::endl;
				out << "d " << material.transparency << std::endl;
				out << "illum 2" << std::endl;
				if (!material.diffuseMap.empty())
					out << "map_Kd " << material.diffuseMap << std::endl;
				if (!material.ambientMap.empty())
					out << "map_Ka " << material.ambientMap << std::endl;
				if (!material.specularMap.empty())
					out << "map_Ks " << material.specularMap << std::endl;
				out << std::endl;
			}
		}

		void OBJExporter::exportOBJ(
			std::ostream& out,
			const std::vector<math::float4>& positions,
			const std::vector<math::float2>& uvs,
			const std::vector<math::float3>& normals,
			const std::vector<Mesh>& meshes,
			bool useLeftHandedCoordsSystem
		)
		{
			exportOBJ(out, positions, uvs, normals, meshes, "", useLeftHandedCoordsSystem);
		}

		void OBJExporter::exportOBJ(
			std::ostream& out, 
			const std::vector<math::float4>& positions, 
			const std::vector<math::float2>& uvs, 
			const std::vector<math::float3>& normals, 
			const std::vector<Mesh>& meshes,
			const std::string& mtllib,
			bool useLeftHandedCoordsSystem
		)
		{
			if (!mtllib.empty())
				out << "mtllib " << mtllib << std::endl;
			
			out << std::fixed << std::setprecision(3);

			std::stringstream tmpOut(std::stringstream::out);
			tmpOut << std::fixed << std::setprecision(3);

			tmpOut << "# " << positions.size() << " vertex positions" << std::endl;
			for (auto i = 0; i < positions.size(); i++)
			{
				const auto& position = positions[i];
				if (useLeftHandedCoordsSystem)
					tmpOut << "v " << position.x << " " << position.y << " " << position.z << std::endl;
				else
					tmpOut << "v " << position.x << " " << position.y << " " << -position.z << std::endl;
			}
			tmpOut << std::endl;

			out << tmpOut.str();
			tmpOut.str(std::string());

			tmpOut << "# " << uvs.size() << " UV coordinates" << std::endl;
			for (auto i = 0; i < uvs.size(); i++)
			{
				const auto& uv = uvs[i];
				tmpOut << "vt " << uv.x << " " << uv.y << " 0" << std::endl;
			}
			tmpOut << std::endl;

			out << tmpOut.str();
			tmpOut.str(std::string());

			tmpOut << "# " << normals.size() << " vertex normals" << std::endl;
			for (auto i = 0; i < normals.size(); i++)
			{
				const auto& normal = normals[i];
				if (useLeftHandedCoordsSystem)
					tmpOut << "vn " << normal.x << " " << normal.y << " " << normal.z << std::endl;
				else
					tmpOut << "vn " << normal.x << " " << normal.y << " " << -normal.z << std::endl;
			}
			tmpOut << std::endl;

			out << tmpOut.str();
			tmpOut.str(std::string());

			for (auto i = 0; i < meshes.size(); i++)
			{
				const auto& mesh = meshes[i];
				// TODO: improve!!!
				if (mesh.indices.size() % 3 != 0)
					throw std::runtime_error("PGA::Rendering::OBJExporter::exportOBJ(): mesh.indices.size() % 3 != 0");
				tmpOut << "# Mesh '" << mesh.name << "' with " << (mesh.indices.size() / 3) << " faces" << std::endl;
				tmpOut << "g " << mesh.name << std::endl;
				if (!mesh.material.empty())
					tmpOut << "usemtl " << mesh.material << std::endl;
				for (auto j = 0; j < mesh.indices.size(); j += 3)
				{
					auto i0 = mesh.indices[j] + 1;
					auto i1 = mesh.indices[j + 1] + 1;
					auto i2 = mesh.indices[j + 2] + 1;
					tmpOut << "f " << i0 << "/" << i0 << "/" << i0 << " " <<
						i1 << "/" << i1 << "/" << i1 << " " <<
						i2 << "/" << i2 << "/" << i2 << " " <<
						std::endl;
				}
				tmpOut << std::endl;
			}
			out << tmpOut.str();
		}

	}

}