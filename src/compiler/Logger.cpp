#include <pga/compiler/Logger.h>

#include <stdarg.h>
#include <cstdio>
#include <string>
#include <map>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace PGA
{
	namespace Compiler
	{
		Logger::Logger()
		{
		}

		Logger::Logger(const Logger& other)
		{
			for (size_t i = 0; i < LL_ALL; i++)
			{
				streams[i].str("");
				streams[i] << other.streams[i].str();
			}
		}

		Logger& Logger::operator=(const Logger& other)
		{
			for (size_t i = 0; i < LL_ALL; i++)
			{
				streams[i].str("");
				streams[i] << other.streams[i].str();
			}
			return *this;
		}

		void Logger::clear()
		{
			for (auto& stream : streams)
				stream.str("");
		}

		bool Logger::hasMessages(Level level)
		{
			return (!streams[level].str().empty());
		}

		void Logger::addMessage(Level level, const char* format, ...)
		{
			va_list args;
			va_start(args, format);
			char buffer[MAX_MESSAGE_SIZE];
#if defined(_WIN32)
			vsprintf_s(buffer, format, args);
#else
			vsprintf(buffer, format, args);
#endif
			va_end(args);
			if (streams[level].str().size() > 0)
				streams[level] << DELIMITER;
			streams[level] << buffer;
		}

		std::stringstream& Logger::operator[](Level level)
		{
			return streams[level];
		}

		const std::string Logger::DELIMITER = ";";

	}

}
