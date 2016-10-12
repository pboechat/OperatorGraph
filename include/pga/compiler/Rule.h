#pragma once

#include <pga/compiler/Successor.h>

#include <string>
#include <memory>

namespace PGA
{
	namespace Compiler
	{
		struct Rule
		{
			std::string symbol;
			std::shared_ptr<Successor> successor;
			bool containsIfCollide;
			
			Rule() : successor(0), containsIfCollide(false) {}
			Rule(const std::string& symbol) : symbol(symbol), successor(0), containsIfCollide(false) {}
		
		};

	};

};
