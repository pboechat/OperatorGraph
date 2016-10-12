#pragma once


__device__ volatile float BigData[1024*1024];

template<int ITS, int REGS = 16>
class DelayClock
{
public:
  __device__ __inline__
  static void delay()
  {
    clock_t t = clock();
    __threadfence_block();

    float values[REGS-2];
    #pragma unroll
    for(int r = 0; r < REGS; ++r)
      values[r] = BigData[threadIdx.x+r*32];
    __threadfence_block();
    #pragma unroll
    for(int r = 0; r < REGS; ++r)
        values[r] += values[r]*values[r];
    __threadfence_block();
    
    while(true)
    {
      clock_t diff = clock() - t;
      if(diff > ITS)
        break;
    }
    #pragma unroll
    for(int r = 0; r < REGS; ++r)
       BigData[threadIdx.x+r*32] =  values[r];
  }
  static std::string name() { return "DelayClock";/*+std::to_string((unsigned long long)ITS);*/ }
  static std::string performanceName() { return "available Giga Cycles (" + std::to_string((long long)REGS) + "regs)";}
  static double performancePerSec(int executedThreads, double s) { return 1.0*executedThreads*ITS/1000000000.0/s; }
};
template<int ITS>
class DelayClock<ITS,0>
{
public:
  __device__ __inline__
  static void delay()
  {
  }
  static std::string name() { return "DelayNone"; }
  static std::string performanceName() { return "Nothing"; }
  static double performancePerSec(int executedThreads, double s) { return 0.0; }
};

template<int ITS, int REGS = 16>
class DelayFMADS
{
public:
  __device__ __inline__
  static void delay()
  {
    float values[REGS];
    #pragma unroll
    for(int r = 0; r < REGS; ++r)
      values[r] = BigData[threadIdx.x+r*32];
    #pragma unroll
    for(int i = 0; i < (ITS+REGS-1)/REGS; ++i)
    {
      #pragma unroll
      for(int r = 0; r < REGS; ++r)
         values[r] += values[r]*values[r];
      __threadfence_block();
    }
    #pragma unroll
    for(int r = 0; r < REGS; ++r)
       BigData[threadIdx.x+r*32] =  values[r];
  }
  static std::string name() { return "DelayFMADS";/*+std::to_string((unsigned long long)ITS);*/ }
  static std::string performanceName() { return "GFLOPS (" + std::to_string((long long)REGS) + "regs)";}
  static double performancePerSec(int executedThreads, double s) { return 2.0*executedThreads*ITS/1000000000.0/s; }
};

template<int ITS>
class DelayFMADS<ITS,0>
{
public:
  __device__ __inline__
  static void delay()
  {
  }
  static std::string name() { return "DelayNone"; }
  static std::string performanceName() { return "Nothing"; }
  static double performancePerSec(int executedThreads, double s) { return 0.0; }
};


template<int ITS, int REGS = 16>
class DelayMem
{
public:
  __device__ __inline__
  static void delay()
  {
    float values[REGS];
    #pragma unroll
    for(int r = 0; r < REGS; ++r)
      values[r] = BigData[threadIdx.x+r*32];
    #pragma unroll
    for(int i = 0; i < (ITS-1+REGS-1)/REGS; ++i)
    {
      #pragma unroll
      for(int r = 0; r < REGS; ++r)
        BigData[threadIdx.x+r*32] = values[r]*values[r];
      __threadfence_block();  
    }
  }
  static std::string name() { return std::string("DelayMem");/*+std::to_string((unsigned long long)ITS);*/ }
  static std::string performanceName() { return "transfer rate GB/s(" + std::to_string((long long)REGS) + "regs)";}
  static double performancePerSec(int executedThreads, double s) { return 4.0*executedThreads*ITS/1000000000.0/s; }
};

template<int ITS>
class DelayMem<ITS,0>
{
public:
  __device__ __inline__
  static void delay()
  {
  }
  static std::string name() { return "DelayNone"; }
  static std::string performanceName() { return "Nothing"; }
  static double performancePerSec(int executedThreads, double s) { return 0.0; }
};