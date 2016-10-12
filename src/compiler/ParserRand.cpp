#include "ParserRand.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(Rand rand)
			{
				return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Rand(rand.min, rand.max));
			}

		}

	}

}