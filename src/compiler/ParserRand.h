#pragma once

#include "ParserElement.h"

#include <pga/compiler/Parameters.h>

#include <memory>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Rand : Element
			{
				double min;
				double max;

			};

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(Rand rand);

		}

	}

}