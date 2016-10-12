#pragma once

#include <memory>

#include <pga/compiler/Parameters.h>
#include "ParserElement.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct ShapeAttribute : Element
			{
				unsigned int type;
				double axis;
				double component;

			};

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const ShapeAttribute& shapeAttribute);

		}

	}

}