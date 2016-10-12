#pragma once

#include <iostream>

#include "utils.h"
#include "types.h"
#include "cuda_memory.h"
#include "timing.h"
#include "delay.cuh"
#include "queueInterface.cuh"
#include "procinfoTemplate.cuh"

#include "techniqueInterface.h"

namespace DynamicParallelism
{
  __host__ __device__ __inline__ uint divup(uint a, uint b)
  {
    return (a + b - 1)/b;
  }


  template<class InitProc, class Q>
  __global__ void initData(Q* q, int num, int frame)
  {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    for( ; id < num; id += blockDim.x*gridDim.x)
    {
      InitProc::template init<Q>(q, id, frame);
    }
  }

  template<class Q>
  __global__ void recordData(Q* q)
  {
    q->record();
  }
  template<class Q>
  __global__ void resetData(Q* q)
  {
    q->reset();
  }

  template<class PROC, class CUSTOM>
  __host__ __device__ void launchKernel(typename PROC::ExpectedData data)
  {

    int nThreads = PROC::NumThreads;
    if(nThreads == 0) 
      nThreads = PROC::ItemInput ? 1 : 256; 
   
    #if !defined(__CUDACC__) || __CUDA_ARCH__ >= 350
    //printf("launching %d\n",PROC::ProcedureId);
    executeProc<PROC, CUSTOM><<<1,divup(nThreads,32)*32, PROC::sharedMemory>>>(data);
    #endif
  }

  class DynQueue : public Queue<>
  {
  public:
  template<class PROCEDURE>
    __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
    {
      launchKernel<PROCEDURE>(data);
      return true;
    }

    template<int threads, class PROCEDURE>
    __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
    {
      if(threadIdx.x % threads == 0)
        enqueue<PROCEDURE>(*data);
      return true;
    }
  };

  template<class PROC, class CUSTOM>
  __global__ void executeProc(typename PROC::ExpectedData data)
  {
    extern __shared__ uint s_data[];
    int num = PROC::NumThreads;
    if(num == 0)
    {
      if(PROC::ItemInput)
        num = 1;
      else
        num = blockDim.x;
    }
    if(threadIdx.x < num)
      PROC:: template execute<DynQueue, Context<PROC::NumThreads, false, CUSTOM> >(threadIdx.x, num, nullptr, &data, s_data);
  }


  //#define PROCCASE(PROCID) \
  //case PROCINFO:: Procedure ## PROCID :: ProcedureId: \
  //nThreads = PROCINFO:: Procedure ## PROCID :: NumThreads; \
  //if(nThreads == 0) \
  //  nThreads = PROCINFO:: Procedure ## PROCID :: ItemInput ? 1 : 256; \
  //if(threadIdx.x % nThreads == 0) \
  //  launchKernel<typename PROCINFO:: Procedure ## PROCID>(*reinterpret_cast<typename PROCINFO:: Procedure ## PROCID::ExpectedData*>(execData)); \
  //break; \


  template<class ProcInfo>
  struct InitLaunchVisitor
  {
    int procId;
    void* execData;

    __device__ __inline__ InitLaunchVisitor(int pId, void* data) : procId(pId), execData(data) { }

    template<class TProcedure, class CUSTOM>
    __device__ __inline__ bool visit()
    {
      if(findProcId<ProcInfo,TProcedure>::value == procId)
      {
        int nThreads = TProcedure::NumThreads == 0 ? ( Procedure::ItemInput? 1 : 256) : TProcedure::NumThreads;
        if(threadIdx.x % nThreads == 0)
          launchKernel<TProcedure, CUSTOM>(*reinterpret_cast<typename TProcedure::ExpectedData*>(execData)); 
        return true;
      }
      return false;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM>
  __global__ void initLaunch(Q* q, uint4 sharedMemDist, int timeLimit)
  {
#if __CUDA_ARCH__ < 350
#ifdef __CUDA_ARCH__
    int arch = __CUDA_ARCH__;
#else
    int arch = 0;
#endif
    if(blockIdx.x == 0 && threadIdx.x == 0)
      printf("Error: Dynamic Parallelism needs Compute Capability 3.5 or higher (using %d)\n", arch);
#else
    __shared__ volatile int runState;
    extern __shared__ uint s_data[];
    void* execData = reinterpret_cast<void*>(s_data + sharedMemDist.x);
    int* execproc = reinterpret_cast<int*>(s_data);

    runState = 1;
    while(runState)
    {
      __syncthreads();
      runState = q-> template dequeue<false> (execData, execproc);
      
      __syncthreads();

      if(runState)
      {

        InitLaunchVisitor<PROCINFO, CUSTOM> visitor(*execproc,execData);
        ProcInfoVisitor<PROCINFO, CUSTOM>:: template Visit<InitLaunchVisitor<PROCINFO, CUSTOM> >(visitor);

      }
    }
#endif
  }
//#undef PROCCASE

  template<template <class> class QUEUE, class PROCINFO, class CUSTOM, bool UseQueue = false, bool NoCopy = false>
  class Technique;

  template<template <class> class QUEUE, class PROCINFO, class CUSTOM>
  class Technique<QUEUE, PROCINFO, CUSTOM, false, false>
  {
    friend struct InitPhaseVisitor;
    friend class PhaseVisitor;
  public:

    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  protected:

    struct InitPhaseVisitor
    {
      Technique &technique;
      InitPhaseVisitor(Technique &technique) : technique(technique) { }
      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {

        technique.blockSize[Phase] = TProcInfo::combMaxNumThreads();
        if(technique.blockSize[Phase] == 0 || TProcInfo::ItemizedOnly())
          technique.blockSize[Phase] = max(technique.blockSize[Phase],256);

        //get shared memory requirement
        technique.sharedMem[Phase] = TProcInfo::requiredShared(technique.blockSize[Phase], false);
        technique.sharedMem[Phase].x /= 4;
        technique.sharedMem[Phase].y = technique.sharedMem[Phase].y/4 + technique.sharedMem[Phase].x;
        technique.sharedMem[Phase].z = technique.sharedMem[Phase].z/4 + technique.sharedMem[Phase].y;

        return false;
      }
    };


    class PhaseVisitor
    {
      Technique& technique;
      int execPhase;
      double execT;

    public:
      PhaseVisitor(Technique& t, int phase, double timeLimitInKCycles) : technique(t), execPhase(phase) { }

      template<class TProcInfo, class TQ, int Phase>
      bool visit()
      {
        if(Phase != execPhase)
          return false;

         PointInTime start;

        //launch initial kernel
        initLaunch<TQ,TProcInfo, CUSTOM><<<4, technique.blockSize[Phase], technique.sharedMem[Phase].y*4>>>(reinterpret_cast<TQ*>(q.get()), technique.sharedMem[Phase]);
        CUDA_CHECKED_CALL(cudaDeviceSynchronize());

        PointInTime end;
      
        execT = end - start;
        return true;
      }

      double getT() { return execT; }
    };



    int blockSize[PROCINFO::NumPhases];
    uint4 sharedMem[PROCINFO::NumPhases];

    std::unique_ptr<Q, cuda_deleter> q;

  public:
    Technique()
    {
    }

    void init()
    {
      int d;
      cudaGetDevice(&d);
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, d);
      if(prop.major*10 + prop.minor < 35)
      {
        std::cout << "Error Dynamic Parallelism requires CC3.5 or higher (only CC"<< prop.major << "." << prop.minor << " available on " << prop.name << std::endl;
        return;
      }

      q = std::unique_ptr<Q, cuda_deleter>(cudaAlloc<Q>());

      initQueue<Q> <<<512, 512>>>(q.get());
      CUDA_CHECKED_CALL(cudaDeviceSynchronize());

      InitPhaseVisitor v(*this);
        Q::template staticVisit<InitPhaseVisitor>(v);

    }

    void resetQueue()
    {
      init();
    }

    void recordQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR DynamicParallelism::recordQueue(): queue does not support reuse init\n";
      else
      {
        recordData<Q><<<1, 1>>>(q.get());
        CUDA_CHECKED_CALL(cudaDeviceSynchronize());
      }
    }

    void restoreQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR DynamicParallelism::restoreQueue(): queue does not support reuse init\n";
      else
        resetData<Q><<<1, 1>>>(q.get());
    }


    template<class InsertFunc>
    void insertIntoQueue(int num)
    {
      typedef CurrentMultiphaseQueue<Q, 0> Phase0Q;

      int b = min((num + 512 - 1)/512,104);
      initData<InsertFunc, Phase0Q><<<b, 512>>>(reinterpret_cast<Phase0Q*>(q.get()), num);
      CUDA_CHECKED_CALL(cudaDeviceSynchronize());
    }

    int BlockSize(int phase = 0) const
    {
      return 0;
    }
    int Blocks(int phase = 0) const
    {
      return 0;
    }
    uint SharedMem(int phase = 0) const
    {
      return 0;
    }

    std::string name() const
    {
      return "DynamicParallelism>" + Q::name();
    }

    void release()
    {
      delete this;
    }

    //exec with our without timelimit
    double execute(int phase = 0, double timelimitInMs = 0)
    {
      if(timelimitInMs != 0)
      {
        std::cout << "ERROR dynamic parallelism does not support a time limit!\n";
        return 0;
      }
      PhaseVisitor v(*this, phase);
      Q::template staticVisit<PhaseVisitor>(v);

      return v.getT();
    }

    template<int Phase, int TimeLimitInKCycles>
    double execute()
    {
      return execute(Phase, TimeLimitInKCycles/static_cast<double>(this->freq)*1000);
    }

    template<int Phase>
    double execute()
    {
      return execute(Phase, 0);
    }

  };


  __device__ int inFlightBlocks = 0;


  template<class PROC, class CUSTOM, class Q, bool NoCopy>
  __global__ void executeProcPull(Q* q, int pullElements)
  {
    extern __shared__ uint s_data[];
    uint* sdata = (uint*)((((unsigned long long)s_data)+15)/16*16);
    int elements = (pullElements + gridDim.x - 1) / gridDim.x;
    elements = max(0,min(elements, pullElements - elements * blockIdx.x));

    
    if(elements > 0)
    {
      void* execData = nullptr;
      int id =  q-> template startRead<PROC> ( execData, elements);
      __syncthreads();

      const int threads = getThreadCount<PROC>();
      if(NoCopy)
      {
        if(threadIdx.x < threads*elements)
          PROC:: template execute<Q, Context<PROC::NumThreads, false, CUSTOM> >(threadIdx.x, threads*elements, q, reinterpret_cast<typename PROC::ExpectedData*>(execData), sdata);
      }
      else
      {
        if(threadIdx.x < elements*threads)
        {
          typename PROC::ExpectedData* pdata = reinterpret_cast<typename PROC::ExpectedData*>(execData);
          *(typename PROC::ExpectedData*)(sdata + sizeof(typename PROC::ExpectedData)/sizeof(uint)*getThreadOffset<PROC, false>()) = *pdata;
        }

        if(threadIdx.x < threads*elements)
          PROC:: template execute<Q, Context<PROC::NumThreads, false, CUSTOM> >(threadIdx.x, threads*elements, q, reinterpret_cast<typename PROC::ExpectedData*>(sdata + threadIdx.x/threads*sizeof(typename PROC::ExpectedData)/sizeof(uint)), sdata + sizeof(typename PROC::ExpectedData)/sizeof(uint)*elements);    
      }
      __syncthreads();
      q-> template finishRead<PROC>(id, elements);
    }
    if(threadIdx.x == 0)
      atomicSub(&inFlightBlocks, 1);
  }

  template<class ProcInfo, class PROC, class CUSTOM, class Q, bool ExecStats, bool NoCopy>
  __device__ int launchDynKernel(Q *q, int& elements, cudaStream_t stream, int2 &execStat, bool fullonly)
  {
#if __CUDA_ARCH__ >= 350
    if(elements != 0)
    { 
      int num = q-> template reserveRead<PROC>( elements );
      if(num == 0)
        return 0;
      int nThreads = PROC::NumThreads;
      if(nThreads == 0) 
        nThreads = PROC::ItemInput ? 1 : 256; 
   
      int blockSize = nThreads;
      int blocks = 1;
      int smem = 16; 
      if(!PROC::ItemInput) 
      { 
        blocks = num;
        smem = sizeof(PROC::ExpectedData) + PROC::sharedMemory;
      }
      else
      {
        blockSize = nThreads*num;
        if(blockSize > 256)
          blockSize = 256 / nThreads * nThreads;
        blocks = (num * nThreads + blockSize - 1) / blockSize;
        if(NoCopy)
          smem = (blockSize/nThreads)*PROC::sharedMemory;
        else
          smem = (sizeof(PROC::ExpectedData)*(blockSize/nThreads) + 15)/16*16 + (blockSize/nThreads)*PROC::sharedMemory;
      }
        
      //printf("launching %d %d for proc: %d\n",blocks, blockSize, PROC::ProcedureId);
      if(ExecStats)
      {
        execStat.x += blocks;
        execStat.y +=num;
      }
      //printf("executing %d with %dx%d smem: %d \n",findProcId<ProcInfo,PROC>::value, blocks,blockSize, smem);
      executeProcPull<PROC, CUSTOM, Q, NoCopy><<<blocks, blockSize, smem+16, stream>>>(q, num);
      atomicAdd(&inFlightBlocks, blocks);
      return blocks;  
    }
#endif
    return 0;
  }
  
  template<class ProcInfo, class Q, bool ExecStats, bool NoCopy>
  struct LaunchVisitor
  {
    bool fullonly;
    int launched;
    int i;
    int* queueCounts;
    int2* execStats;
    cudaStream_t* streams;
    Q* q;

    __device__ __inline__ LaunchVisitor(int* _queueCounts, int2* _execStats, cudaStream_t* _streams, Q* queue) : fullonly(true), launched(0), i(0), queueCounts(_queueCounts), execStats(_execStats), streams(_streams), q(queue) { }

    __device__ __inline__ bool runSecondPass()
    {
      if(launched == 0)
      {
        fullonly = false;
        i = 0;
        return true;
      }
      return false;
    }
    template<class TProcedure, class CUSTOM>
    __device__ __inline__ bool visit()
    {
      launched += launchDynKernel<ProcInfo, TProcedure, CUSTOM, Q, ExecStats, NoCopy>(q, queueCounts[i], streams[i], execStats[i], fullonly);
      ++i;
      return false;
    }
  };

  template<class PROCINFO, class CUSTOM, class Q, bool ExecStats, bool NoCopy, bool SupportTimelimit>
  __global__ void dynamicParallelismController(Q* q, int timeLimitInKCycles)
  {
#if __CUDA_ARCH__ < 350
#ifdef __CUDA_ARCH__
    int arch = __CUDA_ARCH__;
#else
    int arch = 0;
#endif
    if(blockIdx.x == 0 && threadIdx.x == 0)
      printf("Error: Dynamic Parallelism needs Compute Capability 3.5 or higher (using %d)\n", arch);

#else
    __shared__ int queueCounts[PROCINFO::NumProcedures];
    __shared__ int2 execStats[PROCINFO::NumProcedures];
    __shared__ cudaStream_t streams[PROCINFO::NumProcedures];
    __shared__ unsigned long long startTime;

    startTime = clock64();

    if(threadIdx.x < PROCINFO::NumProcedures)
    {
      queueCounts[threadIdx.x] = 0;
      execStats[threadIdx.x] = make_int2(0,0);
      cudaStreamCreateWithFlags(streams + threadIdx.x, cudaStreamNonBlocking);
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
      int lastLaunched = 0;
      int zerolaunches = 0;
      while(true)
      {
        q->numEntries(queueCounts);
        

        LaunchVisitor<PROCINFO, Q, ExecStats, NoCopy> visitor(queueCounts,execStats,streams,q);
        ProcInfoVisitor<PROCINFO, CUSTOM>:: template Visit<LaunchVisitor<PROCINFO, Q, ExecStats, NoCopy> >(visitor);

        //#define LAUNCHENTRY(ProcNum) \
        //  if(PROCINFO::Procedure ## ProcNum::ProcedureId >= 0) \
        //    launched += launchDynKernel<PROCINFO::Procedure ## ProcNum, Q, ExecStats, NoCopy>(q, queueCounts[ProcNum], streams[ProcNum], execStats[ProcNum], fullonly); \

        //  LAUNCHENTRY(0) LAUNCHENTRY(1) LAUNCHENTRY(2) LAUNCHENTRY(3)
        //  LAUNCHENTRY(4) LAUNCHENTRY(5) LAUNCHENTRY(6) LAUNCHENTRY(7)
        //  LAUNCHENTRY(8) LAUNCHENTRY(9) LAUNCHENTRY(10) LAUNCHENTRY(11)
        //  LAUNCHENTRY(12) LAUNCHENTRY(13) LAUNCHENTRY(14) LAUNCHENTRY(15)

        if(visitor.runSecondPass())
        {
          ProcInfoVisitor<PROCINFO, CUSTOM>:: template Visit<LaunchVisitor<PROCINFO, Q, ExecStats, NoCopy> >(visitor);
        }
       /* if(launched == 0)
        {
          fullonly = false;
          LAUNCHENTRY(0) LAUNCHENTRY(1) LAUNCHENTRY(2) LAUNCHENTRY(3)
          LAUNCHENTRY(4) LAUNCHENTRY(5) LAUNCHENTRY(6) LAUNCHENTRY(7)
          LAUNCHENTRY(8) LAUNCHENTRY(9) LAUNCHENTRY(10) LAUNCHENTRY(11)
          LAUNCHENTRY(12) LAUNCHENTRY(13) LAUNCHENTRY(14) LAUNCHENTRY(15)
        }
        #undef LAUNCHENTRY*/

        int inFlight = *(volatile int*)(&inFlightBlocks);
        if(visitor.launched == 0 && inFlight == 0 && lastLaunched == 0)
        {
         // printf("dyn controller done\n");
          break;
        }

        if(SupportTimelimit && static_cast<int>(clock64() - startTime) > timeLimitInKCycles)
        {
          // printf("dyn controller end due to time\n");
          break;
        }

        //else
        //  printf("still acitve: %d %d %d\n",launched,inFlight,lastLaunched);
        lastLaunched = visitor.launched + inFlight;
      }
    }
    
    __syncthreads();
    // FIXME: commented out for the moment cause it was causing "invalid memory access" error when running PGA with instrumentation ON
     /*if(threadIdx.x < 16)
    {
      cudaStreamDestroy(streams[threadIdx.x]);
    }*/
#endif
  }


  template<template <class> class QUEUE, class PROCINFO, class CUSTOM, bool NoCopy>
  class Technique<QUEUE, PROCINFO, CUSTOM, true, NoCopy>
  {
    friend class PhaseVisitor;

  public:
    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  protected:
    std::unique_ptr<Q, cuda_deleter> q;

    int freq;

    class PhaseVisitor
    {
      Technique& technique;
      int execPhase;
      int timeLimitInKCycles;
      double execT;

    public:
      PhaseVisitor(Technique& t, int phase, int timeLimitInKCycles) : technique(t), execPhase(phase), timeLimitInKCycles(timeLimitInKCycles) { }

      template<class TProcInfo, class TQ, int Phase>
      bool visit()
      {
        if(Phase != execPhase)
          return false;


        PointInTime start;

        if(timeLimitInKCycles != 0)
          dynamicParallelismController<TProcInfo,CUSTOM,TQ,NoCopy,1><<<1,32>>>(reinterpret_cast<TQ*>(q.get()), timeLimitInKCycles);
        else
          dynamicParallelismController<TProcInfo,CUSTOM,TQ,NoCopy,0><<<1,32>>>(reinterpret_cast<TQ*>(q.get()), timeLimitInKCycles);
        CUDA_CHECKED_CALL(cudaDeviceSynchronize());

        PointInTime end;

        execT = end - start;
        return true;        
      }

      double getT() { return execT; }
    };


  public:
    Technique(){}

    void init()
    {
      int d;
      cudaGetDevice(&d);
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, d);
      freq = static_cast<int>(static_cast<unsigned long long>(prop.clockRate)*1000/1024);
      if(prop.major*10 + prop.minor < 35)
      {
        std::cout << "Error Dynamic Parallelism requires CC3.5 or higher (only CC"<< prop.major << "." << prop.minor << " available on " << prop.name << std::endl;
        return;
      }

      q = std::unique_ptr<Q, cuda_deleter>(cudaAlloc<Q>());

      initQueue<Q> <<<512, 512>>>(q.get());
      CUDA_CHECKED_CALL(cudaDeviceSynchronize());
    }

      void resetQueue()
    {
      init();
    }

    void recordQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR DynamicParallelism::recordQueue(): queue does not support reuse init\n";
      else
      {
        recordData<Q><<<1, 1>>>(q.get());
        CUDA_CHECKED_CALL(cudaDeviceSynchronize());
      }
    }

    void restoreQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR DynamicParallelism::restoreQueue(): queue does not support reuse init\n";
      else
        resetData<Q><<<1, 1>>>(q.get());
    }


    template<class InsertFunc>
    void insertIntoQueue(int num)
    {
      typedef CurrentMultiphaseQueue<Q, 0> Phase0Q;

      int b = min((num + 512 - 1)/512,104);
      initData<InsertFunc, Phase0Q><<<b, 512>>>(reinterpret_cast<Phase0Q*>(q.get()), num);
      CUDA_CHECKED_CALL(cudaDeviceSynchronize());
    }

    std::string name() const
    {
      return std::string("DynamicParallelismWithController") + (NoCopy?"NoCopy":"")  + ">" +  Q::name();
    }

    void release()
    {
      delete *this;
    }



    //exec with our without timelimit
    double execute(int phase = 0, double timelimitInMs = 0)
    {
      
      PhaseVisitor v(*this, phase, timelimitInMs/1000*freq);
      Q::template staticVisit<PhaseVisitor>(v);

      return v.getT();
    }

    template<int Phase, int TimeLimitInKCycles>
    double execute()
    {
      return execute(Phase, TimeLimitInKCycles/static_cast<double>(freq)*1000);
    }

    template<int Phase>
    double execute()
    {
      return execute(Phase, 0);
    }

  };

  template<template <class> class QUEUE, class PROCINFO, class CUSTOM = void>
  class TechniqueStandard : public Technique<QUEUE, PROCINFO, CUSTOM,  false, false> { };
  template<template <class> class QUEUE, class PROCINFO, class CUSTOM = void>
  class TechniqueQueued : public Technique<QUEUE, PROCINFO, CUSTOM, true, false> { };
  template<template <class> class QUEUE,  class PROCINFO, class CUSTOM = void>
  class TechniqueQueuedNoCopy : public Technique<QUEUE,  PROCINFO, CUSTOM, true, true> { };
}
