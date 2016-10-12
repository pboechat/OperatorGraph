#pragma once

#ifndef WIN32
	#include <sys/time.h>
#else
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

#ifndef WIN32
  class PointInTime
  {
     struct timeval tv;
  public:
    PointInTime()
    {
      gettimeofday(&tv,0);
    }
    double operator - (const PointInTime& o) const
    {
      return (tv.tv_sec-o.tv.tv_sec) + (tv.tv_usec - o.tv.tv_usec) / 1000000.0;
    }
  };
#else
  class PointInTime
  {
    unsigned __int64 t;
    static double perfCounterFreq(bool &hasPerf)
    {
      LONGLONG f;
      hasPerf = QueryPerformanceFrequency((LARGE_INTEGER*)&f) != 0;
      if(hasPerf)
        return static_cast<double>(f);
      else
        return 1000.0;
    }
  public:
    PointInTime()
    {
      static bool has_pc;
      static double freq = perfCounterFreq(has_pc);
      if(has_pc)
      {
        LONGLONG val;
        QueryPerformanceCounter((LARGE_INTEGER*)&val);
        t = val;
      }
      else
        t = GetTickCount();
    }
    double operator - (const PointInTime& o) const
    {
      static bool has_pc;
      static double freq = perfCounterFreq(has_pc);
      return (t - o.t)/freq;
    }
  };
#endif