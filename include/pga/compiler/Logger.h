#pragma once

#include <string>
#include <sstream>

namespace PGA
{
	namespace Compiler
	{
		struct Logger
		{
			enum Level
			{
				LL_ERROR,
				LL_WARNING,
				LL_ALL

			};

			Logger();
			Logger(const Logger& other);

			void clear();
			bool hasMessages(Level level);
			void addMessage(Level level, const char* format, ...);

			Logger& operator=(const Logger& other);
			std::stringstream& operator[](Level level);

		private:
			const static size_t MAX_MESSAGE_SIZE = 4096;
			const static std::string DELIMITER;
			std::stringstream streams[LL_ALL + 1];

		};

	}

}