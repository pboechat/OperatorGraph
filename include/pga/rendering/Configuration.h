#pragma once

#include <vector>
#include <map>
#include <string>

#include <math/vector.h>

namespace PGA
{
	namespace Rendering
	{
		class Configuration
		{
		public:
			struct Material
			{
				unsigned int type;
				bool backFaceCulling;
				int blendMode;
				std::map<std::string, std::string> attributes;

				Material() : type(0), backFaceCulling(true), blendMode(-1) {}

				std::string getAttribute(const std::string& attributeName) const;
				float getFloatAttribute(const std::string& attributeName) const;
				math::float2 getFloat2Attribute(const std::string& attributeName) const;
				math::float4 getFloat4Attribute(const std::string& attributeName) const;
				bool hasAttribute(const std::string& attributeName) const;

			};

			struct Buffer
			{
				std::string name;
				int materialRef;

				Buffer() : name(""), materialRef(-1) {}

			};

			struct TriangleMesh : Buffer
			{
				unsigned int maxNumVertices;
				unsigned int maxNumIndices;

				TriangleMesh() : maxNumVertices(0), maxNumIndices(0) {}

			};

			struct InstancedTriangleMesh : Buffer
			{
				enum Type
				{
					SHAPE,
					OBJ

				};

				unsigned int maxNumElements;
				Type type;
				std::string shape;
				std::string modelPath;
				std::vector<math::float2> vertices;

				InstancedTriangleMesh() : maxNumElements(0), shape("Box"), modelPath("") {}

			};

			std::string modelRootPath;
			std::string textureRootPath;

			std::map<unsigned int, TriangleMesh> triangleMeshes;
			std::map<unsigned int, InstancedTriangleMesh> instancedTriangleMeshes;
			std::map<int, Material> materials;

			void reset();
			void loadFromString(const std::string& xmlString);
			void loadFromDisk(const std::string& fileName);

		};

	}

}