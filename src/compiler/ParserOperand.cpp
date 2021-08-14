#include "ParserExpression.h"
#include "ParserOperand.h"

#include <boost/variant/get.hpp>

#include <stdexcept>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Operand& operand)
			{
				auto type = operand.which();
				switch (type)
				{
				case 0:
					return toParameter(boost::get<Axis>(operand));
				case 1:
					return toParameter(boost::get<RepeatMode>(operand));
				case 2:
					return toParameter(boost::get<ShapeAttribute>(operand));
				case 3:
					return toParameter(boost::get<Vec2>(operand));
				case 4:
					return toParameter(boost::get<Rand>(operand));
				case 5:
					return toParameter(boost::get<Expression>(operand));
				case 6:
					return toParameter(boost::get<double>(operand));
				default:
					throw std::runtime_error("PGA::Compiler::Parser::toParameter(): unknown operand type");
				}
			}

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(double doubleValue)
			{
				return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Scalar(doubleValue));
			}

		}

	}

}