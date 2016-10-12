#pragma once

#include <string>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		struct Terminal
		{
			//std::string path;
			unsigned int idx;
			std::vector<std::string> symbols;
			std::vector<double> parameters;
		};

	}

}
