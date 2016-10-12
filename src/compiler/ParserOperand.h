#pragma once

#include <memory>

#include <boost/variant/recursive_variant.hpp>

#include <pga/compiler/Parameters.h>

#include "ParserAxis.h"
#include "ParserRepeatMode.h"
#include "ParserShapeAttribute.h"
#include "ParserVec2.h"
#include "ParserRand.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Expression;

			typedef boost::variant <
				Axis,
				RepeatMode,
				ShapeAttribute,
				Vec2,
				Rand,
				boost::recursive_wrapper<Expression>,
				double
			> Operand;

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Operand& operand);
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(double doubleValue);

		}

	}

}