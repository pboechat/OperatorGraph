#pragma once

#include <memory>

#include <pga/compiler/Parameters.h>
#include "ParserElement.h"
#include "ParserOperand.h"

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