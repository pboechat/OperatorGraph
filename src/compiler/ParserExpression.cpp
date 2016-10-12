#include <boost/variant/get.hpp>

#include <pga/core/ParameterType.h>

#include "ParserOperand.h"
#include "ParserExpression.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Expression& expression)
			{
				return std::shared_ptr<PGA::Compiler::Parameter>(
					new PGA::Compiler::Exp(
						static_cast<PGA::OperationType>(expression.operation),
						toParameter(expression.left),
						toParameter(expression.right)
					)
				);
			}

		}

	}

}