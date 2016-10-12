#pragma once

#include <memory>
#include <vector>
#include "utils.h"
#include "cuda_memory.h"
#include <iostream>
#include "timing.h"
#include "delay.cuh"

#include "techniqueInterface.h"

#include "procinfoTemplate.cuh"
#include "queuingMultiPhase.cuh"


namespace SegmentedStorage
{
  void checkReinitStorage();
}

namespace Megakernel
{
  __device__ volatile int doneCounter = 0;
  __device__ volatile int endCounter = 0;

  template<class InitProc, class Q>
  __global__ void initData(Q* q, int num)
  {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    for( ; id < num; id += blockDim.x*gridDim.x)
    {
      InitProc::template init<Q>(q, id);
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
  template<class Q>
  __global__ void clearQueueData(Q* q)
  {
    q->clear();
  }

  

  template<class Q, class ProcInfo, class PROC, class CUSTOM, bool Itemized,  bool MultiElement>
  class FuncCaller;


  template<class Q, class ProcInfo, class PROC, class CUSTOM>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, false, false>
  {
  public:
    __device__ __inline__
    static void call(Q* queue, void* data, int hasData, uint* shared)
    {
      int nThreads;
      if(PROC::NumThreads != 0)
        nThreads = PROC::NumThreads;
      else
        nThreads = blockDim.x;
      if(PROC::NumThreads == 0 || threadIdx.x < nThreads)
        PROC :: template execute<Q, Context<PROC::NumThreads, false, CUSTOM> >(threadIdx.x, nThreads, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
    }
  };

  template<class Q, class ProcInfo, class PROC, class CUSTOM>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, false, true>
  {
  public:
    __device__ __inline__
    static void call(Q* queue, void* data, int hasData, uint* shared)
    {
      
      if(PROC::NumThreads != 0)
      {
        int nThreads;
        nThreads = PROC::NumThreads;
        int tid = threadIdx.x % PROC::NumThreads;
        int offset = threadIdx.x / PROC::NumThreads;
        if(threadIdx.x < hasData)
          PROC :: template execute<Q, Context<PROC::NumThreads, true, CUSTOM> >(tid, nThreads, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared + offset*PROC::sharedMemory/sizeof(uint) );
      }
      else
      {
        PROC :: template execute<Q, Context<PROC::NumThreads, true, CUSTOM> >(threadIdx.x, blockDim.x, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
      }
      
    }
  };

  template<class Q, class ProcInfo, class PROC, class CUSTOM, bool MultiElement>
  class FuncCaller<Q, ProcInfo, PROC, CUSTOM, true, MultiElement>
  {
  public:
    __device__ __inline__
    static void call(Q* queue, void* data, int numData, uint* shared)
    {
      if(threadIdx.x < numData)
        PROC :: template execute<Q, Context<PROC::NumThreads, MultiElement, CUSTOM> >(threadIdx.x, numData, queue, reinterpret_cast<typename PROC::ExpectedData*>(data), shared);
    }
  };

  
  ////////////////////////////////////////////////////////////////////////////////////////
  
  template<class Q, class ProcInfo, bool MultiElement>
  struct ProcCallCopyVisitor
  {
    int* execproc;
    const uint4 & sharedMem;
    Q* q;
    void* execData;
    uint* s_data;
	 int hasResult;
    __inline__ __device__ ProcCallCopyVisitor(Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data) { }
    template<class TProcedure, class CUSTOM>
    __device__ __inline__ bool visit()
    {
      if(*execproc == findProcId<ProcInfo, TProcedure>::value)
      {
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );
          return true;
      }
      return false;
    }
  };

  template<class Q, class ProcInfo, bool MultiElement>
  struct ProcCallNoCopyVisitor
  {
    int* execproc;
    const uint4 & sharedMem;
    Q* q;
    void* execData;
    uint* s_data;
    int hasResult;
    __inline__ __device__ ProcCallNoCopyVisitor(Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data), hasResult(hasResult) { }
    template<class TProcedure, class CUSTOM>
    __device__ __inline__ bool visit()
    {
      if(*execproc == findProcId<ProcInfo, TProcedure>::value)
      {
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );
          int n = TProcedure::NumThreads != 0 ?  hasResult / TProcedure ::NumThreads : (TProcedure ::ItemInput  ? hasResult : 1);
          __syncthreads();
          q-> template finishRead<TProcedure>(execproc[1],  n);
          return true;
      }
      return false;
    }
  };

  #define PROCCALLNOCOPYPART(LAUNCHNUM) \
  template<class Q, class ProcInfo, bool MultiElement> \
  struct ProcCallNoCopyVisitorPart ## LAUNCHNUM \
  { \
    int* execproc; \
    const uint4 & sharedMem; \
    Q* q; \
    void* execData; \
    uint* s_data; \
    int hasResult; \
    __inline__ __device__ ProcCallNoCopyVisitorPart ## LAUNCHNUM  (Q* q, int *execproc, void * execData, uint* s_data, const uint4& sharedMem, int hasResult ) : execproc(execproc), sharedMem(sharedMem), q(q), execData(execData), s_data(s_data), hasResult(hasResult) { }  \
    template<class TProcedure, class CUSTOM>  \
    __device__ __inline__ bool visit()  \
    {  \
      if(*execproc == TProcedure::ProcedureId)  \
      {  \
          FuncCaller<Q, ProcInfo, TProcedure, CUSTOM, TProcedure :: ItemInput, MultiElement>::call(q, execData, hasResult, s_data + sharedMem.x + sharedMem.y + sharedMem.w );   \
          int n = TProcedure::NumThreads != 0 ?  hasResult / TProcedure ::NumThreads : (TProcedure ::ItemInput  ? hasResult : 1); \
          q-> template finishRead ## LAUNCHNUM  <TProcedure>(execproc[1],  n);  \
          return true;  \
      }  \
      return false;   \
    }   \
  };

  PROCCALLNOCOPYPART(1)
  PROCCALLNOCOPYPART(2)
  PROCCALLNOCOPYPART(3)

#undef PROCCALLNOCOPYPART

  __device__ int maxConcurrentBlocks = 0;
  __device__ volatile int maxConcurrentBlockEvalDone = 0;


  template<class Q, bool Maintainer>
  class MaintainerCaller;

  template<class Q>
  class MaintainerCaller<Q,true>
  {
  public:
    static __inline__ __device__ bool RunMaintainer(Q* q)
    {
      
      if(blockIdx.x == 1)
      {
        __shared__ bool run;
        run = true;
        __syncthreads();
        int runs = 0;
        while(run)
        {
          q->globalMaintain();
          __syncthreads();
          if(runs > 10)
          {
            if(endCounter == 0)   
               run = false;
            __syncthreads();
          }
          else
            ++runs;
        }
      }
      return false;
    }
  };
  template<class Q>
  class MaintainerCaller<Q,false>
  {
  public:
    static __inline__ __device__ bool RunMaintainer(Q* q)
    {
      return false;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool CopyToShared, bool MultiElement, bool tripleCall>
  class MegakernelLogics;

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement, bool tripleCall>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, true, MultiElement, tripleCall>
  {
  public:
    static  __device__ __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __shared__ char s_data[];
      void* execData = reinterpret_cast<void*>(s_data + 4*(sharedMemDist.x + sharedMemDist.w));
      int* execproc = reinterpret_cast<int*>(s_data + 4 * sharedMemDist.w);

      int hasResult = q-> template dequeue<MultiElement> (execData, execproc, sizeof(uint)*(sharedMemDist.y + sharedMemDist.z));
      
      __syncthreads();

      if(hasResult)
      {
        ProcCallCopyVisitor<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, reinterpret_cast<uint*>(s_data), sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallCopyVisitor<Q, PROCINFO, MultiElement> >(visitor);
      }
      return hasResult;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, false, MultiElement, false>
  {
  public:
    static  __device__ __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __shared__ char s_data[];
      void* execData = reinterpret_cast<void*>(s_data + 4*(sharedMemDist.x + sharedMemDist.w));
      int* execproc = reinterpret_cast<int*>(s_data + 4*sharedMemDist.w);


      int hasResult = q-> template dequeueStartRead<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      __syncthreads();

      if(hasResult)
      {
        ProcCallNoCopyVisitor<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, reinterpret_cast<uint*>(s_data), sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitor<Q, PROCINFO, MultiElement> >(visitor);
      }
      return hasResult;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool MultiElement>
  class MegakernelLogics<Q, PROCINFO, CUSTOM, false, MultiElement, true>
  {
  public:
    static  __device__ __inline__ int  run(Q* q, uint4 sharedMemDist)
    {
      extern __shared__ char s_data[];
      void* execData = reinterpret_cast<void*>(s_data + 4*(sharedMemDist.x + sharedMemDist.w));
      int* execproc = reinterpret_cast<int*>(s_data + 4*sharedMemDist.w);

      int hasResult = q-> template dequeueStartRead1<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart1<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, reinterpret_cast<uint*>(s_data), sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart1<Q, PROCINFO, MultiElement> >(visitor);      
        return hasResult;
      }

      hasResult = q-> template dequeueStartRead2<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
     
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart2<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart2<Q, PROCINFO, MultiElement> >(visitor);          
        return hasResult;
      }

      hasResult = q-> template dequeueStartRead3<MultiElement> (execData, execproc, sizeof(uint)*sharedMemDist.z);
      
      if(hasResult)
      {
        ProcCallNoCopyVisitorPart3<Q, PROCINFO, MultiElement> visitor(q, execproc, execData, s_data, sharedMemDist, hasResult);
        ProcInfoVisitor<PROCINFO, CUSTOM>::template Visit<ProcCallNoCopyVisitorPart3<Q, PROCINFO, MultiElement> >(visitor);         
      }

      return hasResult;
    }
  };

  template<unsigned long long StaticLimit, bool Dynamic>
  struct TimeLimiter;

  template<>
  struct TimeLimiter<0, false>
  {
    __device__ __inline__ TimeLimiter() { }
    __device__ __inline__ bool stop(int tval)
    {
      return false;
    }
  };

  template<unsigned long long StaticLimit>
  struct TimeLimiter<StaticLimit, false>
  {
    unsigned long long  TimeLimiter_start;
    __device__ __inline__ TimeLimiter() 
    {
      if(threadIdx.x == 0)
        TimeLimiter_start = clock64();
    }
    __device__ __inline__ bool stop(int tval)
    {
      return (clock64() - TimeLimiter_start) > StaticLimit;
    }
  };

  template<>
  struct TimeLimiter<0, true>
  {
    unsigned long long  TimeLimiter_start;
    __device__ __inline__ TimeLimiter() 
    {
      if(threadIdx.x == 0)
        TimeLimiter_start = clock64();
    }
    __device__ __inline__ bool stop(int tval)
    {
      return (clock64() - TimeLimiter_start)/1024 > tval;
    }
  };

  template<class Q, class PROCINFO, class CUSTOM, bool CopyToShared, bool MultiElement, bool Maintainer, class TimeLimiter>
  __global__ void megakernel(Q* q, uint4 sharedMemDist, int t)
  {
    if(q == 0)
    {
      if(maxConcurrentBlockEvalDone != 0)
        return;
      if(threadIdx.x == 0)
        atomicAdd(&maxConcurrentBlocks, 1);
      DelayFMADS<10000,4>::delay();
      __syncthreads();
      maxConcurrentBlockEvalDone = 1;
      __threadfence();
      return;
    }
    __shared__ volatile int runState;

    if(MaintainerCaller<Q, Maintainer>::RunMaintainer(q))
      return;

    __shared__ TimeLimiter timelimiter;

    if(threadIdx.x == 0)
    {
      if(endCounter == 0)
        runState = 0;
      else
      {
        atomicAdd((int*)&doneCounter,1);
        if(atomicAdd((int*)&endCounter,1) == 2597)
          atomicSub((int*)&endCounter, 2597);
        runState = 1;
      }
    }
    q->workerStart();
    __syncthreads();

    while(runState)
    {
      int hasResult = MegakernelLogics<Q, PROCINFO, CUSTOM, CopyToShared, MultiElement, Q::needTripleCall>::run(q, sharedMemDist);
      if(threadIdx.x == 0)
      {
        if(timelimiter.stop(t))
          runState = 0;
        else if(hasResult)
        {
          if(runState == 3)
          {
            //back on working
            runState = 1;
            atomicAdd((int*)&doneCounter,1);
            atomicAdd((int*)&endCounter,1);
          }
          else if(runState == 2)
          {
            //back on working
            runState = 1;
            atomicAdd((int*)&doneCounter,1);
          }
        }
        else
        {
          //RUNSTATE UPDATES
          if(runState == 1)
          {
            //first time we are out of work
            atomicSub((int*)&doneCounter,1);
            runState = 2;
          }
          else if(runState == 2)
          {
            if(doneCounter == 0)
            {
              //everyone seems to be out of work -> get ready for end
              atomicSub((int*)&endCounter,1);
              runState = 3;
            }
          }
          else if(runState == 3)
          {
            int d = doneCounter;
            int e = endCounter;
            //printf("%d %d %d\n",blockIdx.x, d, e);
            if(doneCounter != 0)
            {
              //someone started to work again
              atomicAdd((int*)&endCounter,1);
              runState = 2;
            }
            else if(endCounter == 0)
              //everyone is really out of work
              runState = 0;
          }
        }
      }

      __syncthreads();
      q->workerMaintain();
    }
    q->workerEnd();
  }




  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext = void, int maxShared = 16336, bool LoadToShared = true, bool MultiElement = true, bool StaticTimelimit  = false, bool DynamicTimelimit = false>
  class TechniqueCore
  {
    friend struct InitPhaseVisitor;
  public:

    typedef MultiPhaseQueue< PROCINFO, QUEUE > Q;

  protected:    
    
    std::unique_ptr<Q, cuda_deleter> q;

    int blockSize[PROCINFO::NumPhases];
    int blocks[PROCINFO::NumPhases];
    uint4 sharedMem[PROCINFO::NumPhases];
    uint sharedMemSum[PROCINFO::NumPhases];

    cudaEvent_t a, b;
    PointInTime hStart;

    int freq;

    struct InitPhaseVisitor
    {
      TechniqueCore &technique;
      InitPhaseVisitor(TechniqueCore &technique) : technique(technique) { }
      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        technique.blockSize[Phase] = TProcInfo:: template OptimalThreadCount<MultiElement>::Num;
        
        if(TQueue::globalMaintainMinThreads > 0)
         technique.blockSize[Phase] = max(technique.blockSize[Phase],TQueue::globalMaintainMinThreads);

        uint queueSharedMem = TQueue::requiredShared;

        //get shared memory requirement
        technique.sharedMem[Phase] = TProcInfo:: template requiredShared<MultiElement>(technique.blockSize[Phase], LoadToShared, maxShared - queueSharedMem, false);
        //if(!LoadToShared)
        //  sharedMem.x = 16;
        technique.sharedMem[Phase].x /= 4;
        technique.sharedMem[Phase].y = technique.sharedMem[Phase].y/4;
        technique.sharedMem[Phase].z = technique.sharedMem[Phase].z/4;
     
        //x .. procids
        //y .. data
        //z .. shared mem for procedures
        //w .. sum


        //w ... -> shared mem for queues...
        technique.sharedMemSum[Phase] = technique.sharedMem[Phase].w + queueSharedMem;
        technique.sharedMem[Phase].w = queueSharedMem/4;
        
        if(TQueue::globalMaintainMinThreads > 0)
          technique.sharedMemSum[Phase] = max(technique.sharedMemSum[Phase], TQueue::globalMaintainSharedMemory(technique.blockSize[Phase]));

        //get number of blocks to start - gk110 screwes with mutices...
        int nblocks = 0;
        CUDA_CHECKED_CALL(cudaMemcpyToSymbol(maxConcurrentBlocks, &nblocks, sizeof(int)));
        CUDA_CHECKED_CALL(cudaMemcpyToSymbol(maxConcurrentBlockEvalDone, &nblocks, sizeof(int)));
        megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false, TimeLimiter<StaticTimelimit?1000:0, DynamicTimelimit> > <<<512, technique.blockSize[Phase], technique.sharedMemSum[Phase]>>> (0, technique.sharedMem[Phase], 0);


        CUDA_CHECKED_CALL(cudaDeviceSynchronize());
        CUDA_CHECKED_CALL(cudaMemcpyFromSymbol(&nblocks, maxConcurrentBlocks, sizeof(int)));
        technique.blocks[Phase] = nblocks;
        //std::cout << "blocks: " << blocks << std::endl;
        if(technique.blocks[Phase]  == 0)
          printf("ERROR: in Megakernel confguration: dummy launch failed. Check shared memory consumption?\n");
        return false;
      }
    };


    void preCall(bool syncGPU = false)
    {
      int magic = 2597, null = 0;
      CUDA_CHECKED_CALL(cudaMemcpyToSymbol(doneCounter, &null, sizeof(int)));
      CUDA_CHECKED_CALL(cudaMemcpyToSymbol(endCounter, &magic, sizeof(int)));

      if(syncGPU)
        CUDA_CHECKED_CALL(cudaDeviceSynchronize());
      PointInTime start;
            
      CUDA_CHECKED_CALL(cudaEventRecord(a));
    }

    double postCall(bool syncGPU = false)
    {
      float thist;
      CUDA_CHECKED_CALL(cudaEventRecord(b));
      if(syncGPU)
      {
        cudaError_t err = cudaEventSynchronize(b);
        CUDA_CHECKED_CALL(err);

        PointInTime end;
        if(err != cudaSuccess)
          return -1;
        double hTime = end - hStart;
      }
      else
      {
        CUDA_CHECKED_CALL(cudaEventSynchronize(b));
      }

      CUDA_CHECKED_CALL(cudaEventElapsedTime(&thist, a, b));
      return thist / 1000.0; //end - start;
    }

  public:

    TechniqueCore() : a(0)
    { }

    ~TechniqueCore()
    {
     if(a != 0)
      {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
      }
    }

    void init()
    {
      q.reset();
      q = std::unique_ptr<Q, cuda_deleter>(cudaAlloc<Q>());

      int magic = 2597, null = 0;
      CUDA_CHECKED_CALL(cudaMemcpyToSymbol(doneCounter, &null, sizeof(int)));
      CUDA_CHECKED_CALL(cudaMemcpyToSymbol(endCounter, &magic, sizeof(int)));

      SegmentedStorage::checkReinitStorage();
      initQueue<Q> <<<512, 512>>>(q.get());
      CUDA_CHECKED_CALL(cudaDeviceSynchronize());


      InitPhaseVisitor v(*this);
      Q::template staticVisit<InitPhaseVisitor>(v);

    if (a == 0)
    {
      CUDA_CHECKED_CALL(cudaEventCreate(&a));
      CUDA_CHECKED_CALL(cudaEventCreate(&b));
    }

      cudaDeviceProp props;
      int dev;
      cudaGetDevice(&dev);
      cudaGetDeviceProperties(&props, dev);
      freq = static_cast<int>(static_cast<unsigned long long>(props.clockRate)*1000/1024);
    }
	
	void setBlocks(int numBlocks, int phase = 0)
	{
		blocks[phase] = numBlocks;
	}

    void resetQueue()
    {
      init();
    }

    void recordQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR Megakernel::recordQueue(): queue does not support reuse init\n";
      else
      {
        recordData<Q><<<1, 1>>>(q.get());
        CUDA_CHECKED_CALL(cudaDeviceSynchronize());
      }
    }

    void restoreQueue()
    {
      if(!Q::supportReuseInit)
        std::cout << "ERROR Megakernel::restoreQueue(): queue does not support reuse init\n";
      else
        resetData<Q><<<1, 1>>>(q.get());
    }

    void clearQueue()
    {
      clearQueueData<Q><<<256, 512>>>(q.get());
    }


    template<class InsertFunc>
    void insertIntoQueue(int num)
    {
      typedef CurrentMultiphaseQueue<Q, 0> Phase0Q;


      //Phase0Q::pStart();

      //Phase0Q::CurrentPhaseProcInfo::print();

      int b = min((num + 512 - 1)/512,104);
      initData<InsertFunc, Phase0Q><<<b, 512>>>(reinterpret_cast<Phase0Q*>(q.get()), num);
      CUDA_CHECKED_CALL(cudaDeviceSynchronize());
    }

    int BlockSize(int phase = 0) const
    {
      return blockSize[phase];
    }
    int Blocks(int phase = 0) const
    {
      return blocks[phase];
    }
    uint SharedMem(int phase = 0) const
    {
      return sharedMemSum[phase];
    }

    std::string name() const
    {
      return std::string("Megakernel") + (MultiElement?"Dynamic":"Simple") + (LoadToShared?"":"Globaldata") + ">" + Q::name();
    }

    void release()
    {
      delete this;
    }
  };

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext = void, int maxShared = 16336, bool LoadToShared = true, bool MultiElement = true, bool StaticTimelimit  = false, bool DynamicTimelimit = false>
  class Technique;

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,false> : public TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,false>
  { 
    typedef typename TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, false, false>::Q Q;
    struct LaunchVisitor
    {
      int phase;
      int blocks, blockSize, sharedMemSum;
      uint4 sharedMem;
      Q* q;
      LaunchVisitor(Q* q, int phase, int blocks, int blockSize, int sharedMemSum, uint4 sharedMem) : phase(phase), blocks(blocks), blockSize(blockSize), sharedMemSum(sharedMemSum), sharedMem(sharedMem), q(q) { }

      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        if(phase == Phase)
        {
          megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false, TimeLimiter<false,false> ><<<blocks, blockSize, sharedMemSum>>> (reinterpret_cast<TQueue*>(q), sharedMem, 0);
          return true;
        }
        return false;
      }
    };
  public:
    //double execute(int phase = 0)
	double execute(int phase = 0, double timelimitInMs = 0)
    {
      this->preCall(false);

      LaunchVisitor v(this->q.get(), phase, this->blocks[phase], this->blockSize[phase], this->sharedMemSum[phase], this->sharedMem[phase]);
      Q::template staticVisit<LaunchVisitor>(v);

      return this->postCall(false);
    }
  };


  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,true,false> : public TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,true,false>
  {
    typedef typename TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, true, false>::Q Q;
  public:
    template<int Phase, int TimeLimitInKCycles>
    double execute()
    {
      typedef CurrentMultiphaseQueue<Q, Phase> ThisQ;
      this->preCall(false);
      megakernel<ThisQ, typename ThisQ::CurrentPhaseProcInfo, ApplicationContext, LoadToShared, MultiElement, (ThisQ::globalMaintainMinThreads > 0)?true:false,TimeLimiter<TimeLimitInKCycles,false> ><<<this->blocks[Phase], this->blockSize[Phase], this->sharedMemSum[Phase]>>>(this->q.get(), this->sharedMem[Phase], 0);
      return this->postCall(false);
    }

    template<int Phase>
    double execute()
    {
      return this->execute<Phase, 0>();
    }
  };

  template<template <class> class QUEUE, class PROCINFO, class ApplicationContext, int maxShared, bool LoadToShared, bool MultiElement>
  class Technique<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,true> : public TechniqueCore<QUEUE,PROCINFO,ApplicationContext,maxShared,LoadToShared,MultiElement,false,true>
  {
    typedef typename TechniqueCore<QUEUE, PROCINFO, ApplicationContext, maxShared, LoadToShared, MultiElement, false, true>::Q Q;
    struct LaunchVisitor
    {
      int phase;
      int blocks, blockSize, sharedMemSum;
      uint4 sharedMem;
      int timeLimit;
      Q* q;
      LaunchVisitor(Q* q, int phase, int blocks, int blockSize, int sharedMemSum, uint4 sharedMem, int timeLimit) : phase(phase), blocks(blocks), blockSize(blockSize), sharedMemSum(sharedMemSum), sharedMem(sharedMem), timeLimit(timeLimit), q(q) { }

      template<class TProcInfo, class TQueue, int Phase> 
      bool visit()
      {
        if(phase == Phase)
        {
          megakernel<TQueue, TProcInfo, ApplicationContext, LoadToShared, MultiElement, (TQueue::globalMaintainMinThreads > 0)?true:false,TimeLimiter<false,true> ><<<this->blocks, this->blockSize, this->sharedMemSum>>>(reinterpret_cast<TQueue*>(this->q), this->sharedMem, timeLimit);
          return true;
        }
        return false;
      }
    };
  public:
    double execute(int phase = 0, double timelimitInMs = 0)
    {
      this->preCall(false);

      LaunchVisitor v(this->q.get(),phase, this->blocks[phase], this->blockSize[phase], this->sharedMemSum[phase], this->sharedMem[phase], timelimitInMs/1000*this->freq);
      Q::template staticVisit<LaunchVisitor>(v);

      return this->postCall(false);
    }
  };

  // convenience defines

  template<template <class> class Q, class PROCINFO, class CUSTOM, int maxShared = 16336>
  class SimpleShared : public Technique<Q, PROCINFO, CUSTOM, maxShared, true, false>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, int maxShared = 16336>
  class SimplePointed : public Technique<Q, PROCINFO, CUSTOM, maxShared, false, false>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, int maxShared = 16336>
  class DynamicShared : public Technique<Q, PROCINFO, CUSTOM, maxShared, true, true>
  { };
  template<template <class> class Q, class PROCINFO, class CUSTOM, int maxShared = 16336>
  class DynamicPointed : public Technique<Q, PROCINFO, CUSTOM, maxShared, false, true>
  { };

  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class SimpleShared16336 : public SimpleShared<Q, PROCINFO, CUSTOM, 16336>
  { };

    template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class SimpleShared49000: public SimpleShared<Q, PROCINFO, CUSTOM, 49000>
  { };

  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class SimplePointed24576 : public SimplePointed<Q, PROCINFO, CUSTOM, 24576>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class SimplePointed16336 : public SimplePointed<Q, PROCINFO, CUSTOM, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class SimplePointed12000 : public SimplePointed<Q, PROCINFO, CUSTOM, 12000>
  {  };


  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class DynamicShared16336 : public DynamicShared<Q, PROCINFO, CUSTOM, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class DynamicPointed16336 : public DynamicPointed<Q, PROCINFO, CUSTOM, 16336>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class DynamicPointed12000 : public DynamicPointed<Q, PROCINFO, CUSTOM, 12000>
  {  };
  template<template <class> class Q, class PROCINFO, class CUSTOM = void>
  class DynamicPointed11000 : public DynamicPointed<Q,  PROCINFO, CUSTOM, 11000>
  {  };
}
