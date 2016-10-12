#pragma once

namespace PGA
{
	namespace Compiler
	{
		class LineCounter
		{
		public:
			template<class Iterator>
			static void count(Iterator begin, Iterator error, size_t& line, size_t& column)
			{
				line = 0;
				column = 0;
				while (begin != error)
				{
					if (*begin == '\n')
					{
						++line;
						column = 0;
					}
					else
					{
						++column;
					}
					++begin;
				}
			}
		};

	};

};
