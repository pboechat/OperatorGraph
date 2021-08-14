#pragma once

#include "ParserElement.h"

#include <pga/compiler/Parameters.h>

#include <memory>
#include <vector>

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