
/*
* file created by    Markus Steinberger / steinberger ( at ) icg.tugraz.at
*
* modifications by
*/

#ifndef SOFTSHELL_TOOLS_COMMON_INCLUDED
#define SOFTSHELL_TOOLS_COMMON_INCLUDED

#include "types.h"
#include "utils.h"

namespace Softshell
{

  template<int PSIZE>
  class __PointerEquivalent
  {
  public:
    typedef unsigned int type;
#if defined(__CUDACC__)
    static __device__ inline uint isshared(uint p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.shared t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "r"(p) );
      return res;
    }
    static __device__ inline uint islocal(uint p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.local t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "r"(p) );
      return res;
    }
    static __device__ inline uint isconst(uint p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.const, t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "r"(p) );
      return res;
    }
    static __device__ inline uint isglobal(uint p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.global, t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "r"(p) );
      return res;
    }
#endif
  };
  template<>
  class __PointerEquivalent<8>
  {
  public:
    typedef unsigned long long int type;
#if defined(__CUDACC__)
    static __device__ inline uint isshared(unsigned long long int p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.shared t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "l"(p) );
      return res;
    }
    static __device__ inline uint islocal(unsigned long long int p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.local t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "l"(p) );
      return res;
    }
    static __device__ inline uint isconst(unsigned long long int p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.const t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "l"(p) );
      return res;
    }
    static __device__ inline uint isglobal(unsigned long long int p)
    {
      uint res;
      asm("{\n\t"
          ".reg .pred t1;\n\t"
          "isspacep.global t1, %1;\n\t"
          "selp.u32 	%0, 1, 0, t1;\n\t"
          "}\n\t"
          : "=r" (res) : "l"(p) );
      return res;
    }
#endif
  };
  typedef Softshell::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

 #if defined(__CUDACC__)
  __device__ inline uint laneid()
  {
    uint mylaneid;
    asm("mov.u32 %0, %laneid;" : "=r" (mylaneid));
    return mylaneid;
  }
  //requires ptx isa 1.3
  __device__ inline uint warpid()
  {
    uint mywarpid;
    asm("mov.u32 %0, %warpid;" : "=r" (mywarpid));
    return mywarpid;
  }
  __device__ inline uint nwarpid()
  {
    uint mynwarpid;
    asm("mov.u32 %0, %nwarpid;" : "=r" (mynwarpid));
    return mynwarpid;
  }
  //requires ptx isa 1.3
  __device__ inline uint smid()
  {
    uint mysmid;
    asm("mov.u32 %0, %smid;" : "=r" (mysmid));
    return mysmid;
  }
  
#if __CUDA_ARCH__ < 300
  __device__ inline uint gridid()
  {
    uint mygridid;
    asm("mov.u32 %0, %gridid;" : "=r" (mygridid));
    return mygridid;
  }
#else
  __device__ inline unsigned long long gridid()
  {
    unsigned long long mygridid;
    asm("mov.u64 %0, %gridid;" : "=l" (mygridid));
    return mygridid;
  }
#endif
  //requires ptx isa 2.0 and sm_20
  __device__ inline uint nsmid()
  {
    uint mynsmid;
    asm("mov.u32 %0, %nsmid;" : "=r" (mynsmid));
    return mynsmid;
  }
  //requires ptx isa 2.0 and sm_20
  __device__ inline uint lanemask()
  {
    uint lanemask;
    asm("mov.u32 %0, %lanemask_eq;" : "=r" (lanemask));
    return lanemask;
  }
    //requires ptx isa 2.0 and sm_20
  __device__ inline uint lanemask_le()
  {
    uint lanemask;
    asm("mov.u32 %0, %lanemask_le;" : "=r" (lanemask));
    return lanemask;
  }
    //requires ptx isa 2.0 and sm_20
  __device__ inline uint lanemask_lt()
  {
    uint lanemask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r" (lanemask));
    return lanemask;
  }
    //requires ptx isa 2.0 and sm_20
  __device__ inline uint lanemask_ge()
  {
    uint lanemask;
    asm("mov.u32 %0, %lanemask_ge;" : "=r" (lanemask));
    return lanemask;
  }
    //requires ptx isa 2.0 and sm_20
  __device__ inline uint lanemask_gt()
  {
    uint lanemask;
    asm("mov.u32 %0, %lanemask_gt;" : "=r" (lanemask));
    return lanemask;
  }
  __device__ inline void trap()
  {
    asm("trap;");
  }
  //custom sync
  __device__ inline void syncthreads(uint lock = 0, int num = -1)
  {
    if(num == -1)
    {
      asm volatile ("bar.sync %0;" : : "r"(lock));
    }
    else
    {
      asm volatile ("bar.sync %0, %1;" : : "r"(lock), "r"(num));
    }
  }
  __device__ inline void arrive(uint lock, uint num)
  {
    asm volatile ("bar.arrive %0, %1;" : : "r"(lock), "r"(num));
  }
  __device__ inline uint syncthreads_count(uint predicate, uint lock = 0, int num = -1)
  {
    uint res;
    if(num == -1)
    {
      asm volatile ("bar.red.popc.u32 %0, %1, %2;" : "=r" (res) : "r"(lock), "r" (predicate) );
    }
    else
    {
       asm volatile ("bar.red.popc.u32 %0, %1, %2, %3;" : "=r" (res) : "r"(lock), "r"(num), "r" (predicate) );
    }
    return res;
  }
  __device__ inline int syncthreads_or(int predicate, uint lock = 0, int num = -1)
  {
    int res;
    if(num == -1)
    {
      asm volatile ("bar.red.or.pred %0, %1, %2;" : "=r" (res) : "r"(lock), "r" (predicate) );
    }
    else
    {
       asm volatile ("bar.red.or.pred %0, %1, %2, %3;" : "=r" (res) : "r"(lock), "r"(num), "r" (predicate) );
    }
    return res;
  }
  __device__ inline int syncthreads_and(int predicate, uint lock = 0, int num = -1)
  {
    int res;
    if(num == -1)
    {
      asm volatile ("bar.red.and.pred %0, %1, %2;" : "=r" (res) : "r"(lock), "r" (predicate) );
    }
    else
    {
       asm volatile ("bar.red.and.pred %0, %1, %2, %3;" : "=r" (res) : "r"(lock), "r"(num), "r" (predicate) );
    }
    return res;
  }
  __device__ inline clock_t __clock()
  {
    volatile __shared__ clock_t hack;
    clock_t c;
#ifndef __GNUC__
    asm volatile ("mov.u32 %0, %%clock;" : "=r" (c));
#else
    asm volatile ("mov.u64 %0, %%clock;" : "=l" (c));
#endif
    hack = c;
    return hack;
  }
  __device__ inline uint isshared(void* p)
  {
    return Softshell::__PointerEquivalent<sizeof(char*)>::isshared((PointerEquivalent)p);
  }
  __device__ inline uint islocal(void* p)
  {
    return Softshell::__PointerEquivalent<sizeof(char*)>::islocal((PointerEquivalent)p);
  }
  __device__ inline uint isconst(void* p)
  {
    return Softshell::__PointerEquivalent<sizeof(char*)>::isconst((PointerEquivalent)p);
  }
  __device__ inline uint isglobal(void* p)
  {
    return Softshell::__PointerEquivalent<sizeof(char*)>::isglobal((PointerEquivalent)p);
  }
#endif
  typedef unsigned long long int clock64_t;
#if defined(__CUDACC__)
  __device__ inline clock64_t __clock64()
  {
    volatile __shared__ clock64_t hack;
    clock64_t c;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l" (c));
    hack = c;
    return hack;
  }
#endif

  template<class T>
  __host__ __device__ inline T divup(T a, T b) { return (a + b - 1)/b; }

#if defined(__CUDACC__)
  __device__ inline void wait(uint cycles)
  {
    clock_t start = clock();
    while(clock()-start < cycles);
  }
#endif

#if defined(__CUDACC__)

  template<typename T>
  __device__  inline T atomicAddVolatile(volatile T* p, T val)
  {
    return atomicAdd(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicSubVolatile(volatile T* p, T val)
  {
    return atomicSub(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicExchVolatile(volatile T* p, T val)
  {
    return atomicExch(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicMinVolatile(volatile T* p, T val)
  {
    return atomicMin(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicMaxVolatile(volatile T* p, T val)
  {
    return atomicMax(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicIncVolatile(volatile T* p, T val)
  {
    return atomicInc(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicDecVolatile(volatile T* p, T val)
  {
    return atomicDec(const_cast<T*>(p), val);
  }
  template<typename T>
  __device__  inline T atomicCASVolatile(volatile T* p, T compare, T val)
  {
    return atomicCAS(const_cast<T*>(p), compare, val);
  }

#define __Serialize \
      for (uint __active = 0, \
        __lanemask = lanemask_lt(), \
        __mask = __ballot(1), \
        __local_id = __popc(__lanemask & __mask), \
        __num = __popc(__mask);  \
          \
        __active < __num; \
        \
        ++__active) \
        if (__active == __local_id)
#endif
}

#endif
