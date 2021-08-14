#include "ParserAxis.h"

#include <pga/core/Axis.h>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Axis& axis)
			{
				return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Axis(static_cast<PGA::Axis>(axis.type)));
			}

		}

	}

}