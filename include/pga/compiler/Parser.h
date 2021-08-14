#pragma once

#include <pga/compiler/Axiom.h>
#include <pga/compiler/Logger.h>
#include <pga/compiler/Rule.h>
#include <pga/core/DispatchTableEntry.h>

#include <string>
#include <vector>

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