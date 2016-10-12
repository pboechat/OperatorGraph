#include "ParserVec2.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const Vec2& vector)
			{
				return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Vec2(vector.x, vector.y));
			}

		}

	}

}