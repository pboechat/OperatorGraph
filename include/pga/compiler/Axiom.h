#pragma once

#include <pga/compiler/ShapeType.h>

#include <math/vector.h>

#include <string>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		struct Axiom
		{
			std::string symbol;
			int entryIndex;
			ShapeType shapeType;
			std::vector<math::float2> vertices;

			// TODO:
			Axiom() : symbol("A"), entryIndex(-1), shapeType(BOX) {}

		};

	}

}