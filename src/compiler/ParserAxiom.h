#pragma once

#include <string>
#include <vector>

#include <math/vector.h>

#include "ParserElement.h"

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