#pragma once

#include <string>

#include "ParserElement.h"

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