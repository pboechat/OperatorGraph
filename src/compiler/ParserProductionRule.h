#pragma once

#include "ParserElement.h"
#include "ParserSuccessor.h"
#include "Terminal.h"

#include <pga/compiler/Logger.h>
#include <pga/compiler/Rule.h>

#include <string>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct ProductionRule : Element
			{
				std::string symbol;
				std::vector<Successor> successors;

				void convertToAbstraction(std::vector<PGA::Compiler::Terminal>& terminals, PGA::Compiler::Rule& rule) const;
				bool check(Logger& logger);

			};

		}

	}

}