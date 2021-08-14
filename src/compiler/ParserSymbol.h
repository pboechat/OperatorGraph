#pragma once

#include "ParserElement.h"

#include <string>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Symbol : Element
			{
				Symbol() : symbol(""), probability(0) { }
				std::string symbol;
				double probability;

			};

		}

	}

}