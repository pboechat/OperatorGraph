#pragma once

#include <string>
#include <vector>

#include <pga/compiler/Rule.h>
#include <pga/compiler/Logger.h>

#include "Terminal.h"
#include "ParserElement.h"
#include "ParserSuccessor.h"

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