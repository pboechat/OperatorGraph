#include <pga/core/RepeatMode.h>

#include "ParserRepeatMode.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const RepeatMode& repeatMode)
			{
				return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::RepeatMode(static_cast<PGA::RepeatMode>(repeatMode.type)));
			}

		}

	}

}