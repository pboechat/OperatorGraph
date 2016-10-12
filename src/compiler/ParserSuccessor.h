#pragma once

#include <boost/variant/recursive_variant.hpp>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Symbol;
			struct Operator;

			typedef boost::variant <
				boost::recursive_wrapper<Symbol>,
				boost::recursive_wrapper<Operator>
			> Successor;

		}

	}

}