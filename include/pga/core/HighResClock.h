#pragma once

#include <windows.h>

#include <chrono>

namespace PGA
{
	struct HighResClock
	{
		typedef long long rep;
		typedef std::nano period;
		typedef std::chrono::duration<rep, period> duration;
		typedef std::chrono::time_point<HighResClock> time_point;
		static const bool is_steady = true;
		static time_point now()
		{
#if defined(_WIN32)
			LARGE_INTEGER frequency;
			QueryPerformanceFrequency(&frequency);
			LARGE_INTEGER count;
			QueryPerformanceCounter(&count);
			return time_point(duration(count.QuadPart * static_cast<rep>(period::den) / frequency.QuadPart));
#else
			timespec ts;
			clock_gettime(CLOCK_REALTIME, &ts);
			return time_point(duration(ts.tv_nsec));
#endif
		}

	};

}
