#pragma once

#include "ParserElement.h"

#include <string>
#include <vector>

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