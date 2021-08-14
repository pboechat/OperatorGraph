#pragma once

#include "ParserElement.h"
#include "ParserOperand.h"

#include <pga/compiler/Parameters.h>

#include <memory>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Expression : Element
			{
				unsigned int operation;
				Operand left;
				Operand right;

			};

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Expression& expression);

		}

	}

}