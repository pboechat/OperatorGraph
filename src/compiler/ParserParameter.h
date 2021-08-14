#pragma once

#include "ParserAxis.h"
#include "ParserExpression.h"
#include "ParserRand.h"
#include "ParserRepeatMode.h"
#include "ParserShapeAttribute.h"
#include "ParserVec2.h"

#include <boost/variant/recursive_variant.hpp>
#include <pga/compiler/Parameters.h>

#include <initializer_list>
#include <memory>

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