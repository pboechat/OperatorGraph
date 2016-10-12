#pragma once

#include <vector>
#include <list>

#include "ParserAxiom.h"
#include "ParserProductionRule.h"
#include "ParserTerminal.h"
#include "Terminal.h"
#include <pga/compiler/Axiom.h>
#include <pga/compiler/Rule.h>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Result
			{
				std::list<Axiom> axioms;
				std::list<ProductionRule> rules;
				std::list<Terminal> terminals;

				void convertToAbstraction(std::vector<PGA::Compiler::Axiom>& axioms, std::vector<PGA::Compiler::Rule>& rules, std::vector<PGA::Compiler::Terminal>& terminals) const;
				bool check(Logger& logger);

			};

		}

	}

}