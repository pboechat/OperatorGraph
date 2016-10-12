#pragma once

#include <pga/compiler/ShapeType.h>

#include <vector>

namespace PGA
{
	namespace Compiler
	{
		enum SuccessorType
		{
			OPERATOR,
			SYMBOL

		};

		struct Successor
		{
			ShapeType shapeType;
			size_t id;
			std::string myrule;
			int phase;

			Successor() : phase(-1) {}
			std::vector<double> terminalAttributes;
			virtual ~Successor() {}
			virtual SuccessorType getType() const = 0;

		};

	}

}
