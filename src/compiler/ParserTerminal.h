#pragma once

#include <string>
#include <vector>

#include "ParserElement.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Terminal : Element
			{
				std::vector<std::string> symbols;
				std::vector<double> parameters;

			};

		}

	}

}