#pragma once

#include <cupti.h>

#include <stdexcept>

namespace PGA
{
	namespace CUPTI
	{
		class Exception : public std::exception
		{
		private:
			CUptiResult result;

		public:
			Exception(CUptiResult result) : result(result)
			{
			}

#if defined(_MSC_VER) && _MSC_VER <= 1800
			const char* what() const
#else
			const char* what() const noexcept
#endif
			{
				return "CUPTI fail";
			}
		};

		inline void checkError(CUptiResult result)
		{
			if (result != CUPTI_SUCCESS)
				throw PGA::CUPTI::Exception(result);
		}

	}

}
