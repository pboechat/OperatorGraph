#pragma once

#include <string>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			size_t nextSuccessorIdx();
			std::string ltrim(std::string s);
			std::string rtrim(std::string s);
			std::string trim(std::string s);

		}

	}

}