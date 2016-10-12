#pragma once

#include <pga/compiler/Successor.h>

#include <string>

namespace PGA
{
	namespace Compiler
	{
		struct Symbol : public Successor
		{
			std::string name;			

			Symbol() {}
			Symbol(const std::string& name) : name(name) { }
			virtual ~Symbol() {}
			virtual SuccessorType getType() const { return SuccessorType::SYMBOL; }

		};

	}

}
