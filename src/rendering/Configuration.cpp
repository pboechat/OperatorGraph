#include <fstream>
#include <algorithm>
#include <sstream>
#include <climits>
#include <stdexcept>

#include <rapidxml.h>

#include <pga/core/GeometryUtils.h>
#include <pga/core/StringUtils.h>
#include <pga/rendering/RenderingConstants.h>
#include <pga/rendering/Configuration.h>

namespace
{
	const unsigned int MaxNumImplicitMaterials = 100;

	template <typename T, typename U>
	bool checkKeySpaceContinuity(std::map<T, U>& map)
	{
		int max = -1;
		for (auto& entry : map)
		{
			int curr = static_cast<int>(entry.first);
			if (curr > max)
				max = curr;
		}
		return static_cast<int>(map.size()) == (max + 1);
	}

	int nextImplicitMaterialId
	(
		std::map<int, PGA::Rendering::Configuration::Material>& materials
	)
	{
		int min = 0;
		for (auto& entry : materials)
		{
			if (entry.first < min)
			{
				if (entry.first == std::numeric_limits<int>::min())
					throw std::runtime_error("PGA::Rendering::Configuration::nextImplicitMaterialId(): more implicit materials than allowed [MaxNumImplicitMaterials=" + std::to_string(MaxNumImplicitMaterials) + "]");
				min = entry.first;
			}
		}
		if (min == 0)
			return std::numeric_limits<int>::min() + MaxNumImplicitMaterials;
		else
			return min - 1;
	}

	int parseMaterial
	(
		rapidxml::xml_node<char>* node, 
		std::map<int, PGA::Rendering::Configuration::Material>& materials
	)
	{
		int id = rapidxml::getInt(node, "id", -1);
		id = (id < 0) ? nextImplicitMaterialId(materials) : id;
		PGA::Rendering::Configuration::Material material;
		material.type = rapidxml::getUInt(node, "type", 0);
		material.backFaceCulling = rapidxml::getBool(node, "backFaceCulling", true);
		material.blendMode = rapidxml::getInt(node, "blendMode", -1);
		for (auto* child = node->first_node(); child; child = child->next_sibling())
		{
			if (rapidxml::getName(child) == "Attribute")
				material.attributes[rapidxml::getString(child, "name")] = rapidxml::getString(child, "value");
		}
		materials.emplace(id, material);
		return id;
	}

	void parseInstancedTriangleMesh
	(
		rapidxml::xml_node<char>* node, 
		std::map<unsigned int, PGA::Rendering::Configuration::InstancedTriangleMesh>& instancedTriangleMeshes,
		std::map<int, PGA::Rendering::Configuration::Material>& materials
	)
	{
		auto name = rapidxml::getString(node, "name");
		auto i = rapidxml::getUInt(node, "index");
		// FIXME: checking invariants
		if (i > PGA::Rendering::Constants::MaxNumInstancedTriangleMeshes)
			throw std::runtime_error("PGA::Rendering::Configuration::parseInstancedTriangleMesh(): instanced triangle mesh index is greater than PGA::Rendering::Constants::MaxNumInstancedTriangleMeshes [i=" + std::to_string(i) + "]");
		PGA::Rendering::Configuration::InstancedTriangleMesh instancedTriangleMesh;
		instancedTriangleMesh.name = name;
		instancedTriangleMesh.maxNumElements = rapidxml::getUInt(node, "maxNumElements");
		instancedTriangleMesh.type = static_cast<PGA::Rendering::Configuration::InstancedTriangleMesh::Type>(rapidxml::getUInt(node, "type"));
		instancedTriangleMesh.shape = rapidxml::getString(node, "shape");
		instancedTriangleMesh.modelPath = rapidxml::getString(node, "modelPath");
		auto vec2ListStr = rapidxml::getString(node, "vertices");
		if (!vec2ListStr.empty())
		{
			bool reorder = true;
			auto pos = vec2ListStr.find("(");
			if (pos != std::string::npos)
			{
				auto prefix = vec2ListStr.substr(0, pos);
				if (prefix == "o")
					reorder = false;
			}
			std::vector<math::float2> vec2List;
			PGA::StringUtils::parseVec2List(vec2ListStr, vec2List);
			// FIXME: checking invariants
			if (vec2List.empty())
				throw std::runtime_error("vec2List.empty()");
			if (reorder)
				PGA::GeometryUtils::orderVertices_CCW(vec2List, instancedTriangleMesh.vertices);
			else
				instancedTriangleMesh.vertices.insert(instancedTriangleMesh.vertices.begin(), vec2List.begin(), vec2List.end());
		}
		for (auto* child = node->first_node(); child; child = child->next_sibling())
		{
			if (rapidxml::getName(child) == "MaterialRef")
				instancedTriangleMesh.materialRef = rapidxml::getInt(child, "id", -1);
			else if (rapidxml::getName(child) == "Material")
				instancedTriangleMesh.materialRef = parseMaterial(child, materials);
		}
		instancedTriangleMeshes.emplace(i, instancedTriangleMesh);
	}

	void parseTriangleMesh
	(
		rapidxml::xml_node<char>* node, 
		std::map<unsigned int, PGA::Rendering::Configuration::TriangleMesh>& triangleMeshes,
		std::map<int, PGA::Rendering::Configuration::Material>& materials
	)
	{
		auto name = rapidxml::getString(node, "name");
		auto i = rapidxml::getUInt(node, "index");
		// FIXME: checking invariants
		if (i > PGA::Rendering::Constants::MaxNumTriangleMeshes)
			throw std::runtime_error("PGA::Rendering::Configuration::parseInstancedTriangleMesh(): triangle mesh index is greater then PGA::Rendering::Constants::MaxNumTriangleMeshes [i=" + std::to_string(i) + "]");
		PGA::Rendering::Configuration::TriangleMesh triangleMesh;
		triangleMesh.name = name;
		triangleMesh.maxNumVertices = rapidxml::getUInt(node, "maxNumVertices");
		triangleMesh.maxNumIndices = rapidxml::getUInt(node, "maxNumIndices");
		for (auto* child = node->first_node(); child; child = child->next_sibling())
		{
			if (rapidxml::getName(child) == "MaterialRef")
				triangleMesh.materialRef = rapidxml::getInt(child, "id", -1);
			else if (rapidxml::getName(child) == "Material")
				triangleMesh.materialRef = parseMaterial(child, materials);
		}
		triangleMeshes.emplace(i, triangleMesh);
	}

}

namespace PGA
{
	namespace Rendering
	{
		//////////////////////////////////////////////////////////////////////////
		void Configuration::reset()
		{
			textureRootPath = "";
			modelRootPath = "";
			triangleMeshes.clear();
			instancedTriangleMeshes.clear();
			materials.clear();
		}

		void Configuration::loadFromDisk(const std::string& fileName)
		{
			std::ifstream file(fileName);
			std::string xmlStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
			file.close();
			loadFromString(xmlStr);
		}

		void Configuration::loadFromString(const std::string& xmlStr)
		{
			reset();
			rapidxml::xml_document<> xmlDoc;
			xmlDoc.parse<0>(const_cast<char*>(xmlStr.c_str()));
			auto* root = xmlDoc.first_node();
			if (root && rapidxml::getName(root) == "Configuration")
			{
				for (auto* child = root->first_node(); child; child = child->next_sibling())
				{
					if (rapidxml::getName(child) == "ModelRootPath")
						modelRootPath = rapidxml::getString(child, "path");
					else if (rapidxml::getName(child) == "TextureRootPath")
						textureRootPath = rapidxml::getString(child, "path");
					else if (rapidxml::getName(child) == "Material")
						parseMaterial(child, materials);
					else if (rapidxml::getName(child) == "TriangleMesh")
						parseTriangleMesh(child, triangleMeshes, materials);
					else if (rapidxml::getName(child) == "InstancedTriangleMesh")
						parseInstancedTriangleMesh(child, instancedTriangleMeshes, materials);
				}
			}
			if (!checkKeySpaceContinuity(triangleMeshes))
				throw std::runtime_error("PGA::Rendering::Configuration::loadFromString(): missing triangle mesh indices");
			if (!checkKeySpaceContinuity(instancedTriangleMeshes))
				throw std::runtime_error("PGA::Rendering::Configuration::loadFromString(): missing instanced triangle mesh indices");
		}

		//////////////////////////////////////////////////////////////////////////
		std::string Configuration::Material::getAttribute(const std::string& attributeName) const
		{
			const auto& it = attributes.find(attributeName);
			if (it == attributes.end())
				return "";
			return it->second;
		}

		float Configuration::Material::getFloatAttribute(const std::string& attributeName) const
		{
			const auto& it = attributes.find(attributeName);
			if (it == attributes.end())
				return 0.0f;
			return static_cast<float>(atof(it->second.c_str()));
		}

		math::float2 Configuration::Material::getFloat2Attribute(const std::string& attributeName) const
		{
			const auto& it = attributes.find(attributeName);
			if (it == attributes.end())
				return math::float2();
			return PGA::StringUtils::parseVec2(it->second);
		}

		math::float4 Configuration::Material::getFloat4Attribute(const std::string& attributeName) const
		{
			const auto& it = attributes.find(attributeName);
			if (it == attributes.end())
				return math::float4();
			return PGA::StringUtils::parseVec4(it->second);
		}

		bool Configuration::Material::hasAttribute(const std::string& attributeName) const
		{
			return attributes.find(attributeName) != attributes.end();
		}

	}

}
