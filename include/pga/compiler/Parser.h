#pragma once

#include <string>
#include <vector>

#include <pga/core/DispatchTableEntry.h>
#include <pga/compiler/Axiom.h>
#include <pga/compiler/Rule.h>
#include <pga/compiler/Logger.h>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			bool parse(const std::string& sourceCode, Logger& logger, std::vector<PGA::Compiler::Axiom>& axioms, std::vector<PGA::Compiler::Rule>& rules);

		}

	}

}