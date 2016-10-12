#pragma once

#include <initializer_list>
#include <memory>

#include <boost/variant/recursive_variant.hpp>

#include <pga/compiler/Parameters.h>
#include "ParserAxis.h"
#include "ParserRepeatMode.h"
#include "ParserVec2.h"
#include "ParserShapeAttribute.h"
#include "ParserRand.h"
#include "ParserExpression.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			typedef boost::variant <
				Axis,
				RepeatMode,
				Vec2,
				ShapeAttribute,
				Rand,
				Expression,
				double
			> Parameter;

			std::shared_ptr<PGA::Compiler::Parameter> createParameter(Parameter param);
			std::shared_ptr<PGA::Compiler::Parameter> createParameter(Parameter param,
				const std::initializer_list<int>& allowed,
				const std::initializer_list<int>& forbidden);

		}

	}

}