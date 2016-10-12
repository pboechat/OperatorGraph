#pragma once

#include <vector>
#include <memory>

#include <pga/compiler/Parameters.h>
#include "ParserElement.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Vec2 : Element
			{
				double x;
				double y;

			};

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Vec2& vector);

		}

	}

}