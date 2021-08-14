#pragma once

#include "ParserElement.h"

#include <math/vector.h>

#include <string>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Axiom : Element
			{
				std::string name;
				unsigned int shapeType;
				std::vector<math::float2> vertices;

			};

		}

	}

}