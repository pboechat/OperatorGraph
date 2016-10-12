#pragma once

#include <vector>

#include "ParserElement.h"
#include "ParserOperand.h"
#include "ParserSuccessor.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct ParameterizedSuccessor : Element
			{
				std::vector<Operand> parameters;
				Successor successor;

				ParameterizedSuccessor() {}
				ParameterizedSuccessor(const PGA::Compiler::Parser::Successor& successor) : successor(successor) {}

			};

		}

	}

}