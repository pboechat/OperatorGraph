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
			struct RepeatMode : Element
			{
				unsigned int type;

			};

			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const RepeatMode& repeatMode);

		}

	}

}