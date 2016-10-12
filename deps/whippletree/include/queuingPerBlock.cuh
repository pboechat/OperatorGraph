#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "random.cuh"


template<class ProcedureInfo, template<class ProcedureInfo> class TInternalQueueing, int NumQueues, bool TAssertOnOverflow = true>
class PerBlockQueue : public ::Queue<> 
{
protected:
  typedef TInternalQueueing<ProcedureInfo> InternalQueue;
  InternalQueue queues[NumQueues];

  int dummy[32]; //compiler alignment mismatch hack
public:

  static const bool supportReuseInit = InternalQueue::supportReuseInit;

  static std::string name()
  {
    return std::string("DistributedPerBlock_") + InternalQueue::name();
  }

  __inline__ __device__ void init() 
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].init();
  }

  template<class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    int qId = qrandom::warp_rand() % NumQueues;
    for(int i = 0; i < NumQueues; ++i)
    {
      bool in  = queues[qId].template enqueueInitial<PROCEDURE>(data);
      if(in) return true;
      qId = (qId + 1) % NumQueues;
    }
    if(TAssertOnOverflow)
    {
      printf("ERROR: DistributedPerBlock out of elements!\n");
      Softshell::trap();
    }
    return false;
  }

  template<class PROCEDURE>
  __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {        
    return queues[blockIdx.x % NumQueues]. template enqueue<PROCEDURE>(data);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    if(!queues[blockIdx.x % NumQueues]. template enqueue<threads,PROCEDURE>(data))
    {
      if(TAssertOnOverflow)
      {
        printf("ERROR: DistributedPerBlock out of elements!\n");
        Softshell::trap();
      }
      return false;
    }
    return true;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = 100000)
  {     
    return queues[blockIdx.x % NumQueues] . template dequeue<MultiProcedure>(data,procId,maxShared);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
  {
    return queues[blockIdx.x % NumQueues] . template dequeueSelected<MultiProcedure>(data,procId,maxNum);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = 100000)
  {
    return queues[blockIdx.x % NumQueues] . template dequeueStartRead<MultiProcedure>(data,procId,maxShared);
  }

  template<class PROCEDURE>
  __inline__ __device__ int reserveRead(int maxNum = -1)
  {
    return queues[blockIdx.x % NumQueues] . template reserveRead<PROCEDURE>(maxNum);
  }
  template<class PROCEDURE>
  __inline__ __device__ int startRead(void*& data, int num)
  {
    return queues[blockIdx.x % NumQueues] . template startRead<PROCEDURE>(data, num);
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead(int id,  int num)
  {
    return queues[blockIdx.x % NumQueues] . template finishRead<PROCEDURE>(id, num);
  }

  __inline__ __device__ void numEntries(int* counts)
  { 
    return queues[blockIdx.x % NumQueues] . numEntries(counts);
  }
  
  __inline__ __device__ void workerStart()
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].workerStart();
  }
  __inline__ __device__ void workerMaintain()
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].workerMaintain();
  }
  __inline__ __device__ void workerEnd()
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].workerEnd();
  }
  __inline__ __device__ void globalMaintain()
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].globalMaintain();
  }

  __inline__ __device__ void record()
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].record();
  }

  __inline__ __device__ void reset()
  {
    for(int i = 0; i < NumQueues; ++i)
      queues[i].reset();
  }
};

template<template<class ProcedureInfo> class TInternalQueueing, int NumQueues, bool TAssertOnOverflow = true>
struct PerBlockQueueTyping 
{
  template<class ProcedureInfo>
  class Type : public PerBlockQueue<ProcedureInfo, TInternalQueueing, NumQueues, TAssertOnOverflow> {}; 
};

template<class ProcedureInfo, template<class ProcedureInfo> class TInternalQueueing, int NumQueues, bool TAssertOnOverflow = true>
class PerBlockStealing : public PerBlockQueue<ProcedureInfo, TInternalQueueing, NumQueues, TAssertOnOverflow>
{
public:
  static std::string name()
  {
    return std::string("DistributedPerBlockStealing_") + InternalQueue::name();
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = 100000)
  {     
    int b = blockIdx.x % NumQueues;
    #pragma unroll
    for(int i = 0; i < NumQueues; ++i)
    {
      int d = queues[b] . template dequeue<MultiProcedure>(data, procId, maxShared);
      if(d) return d;
      b = (b + 1) % NumQueues;
    }
    return 0;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
  {
    int b = blockIdx.x % NumQueues;
    #pragma unroll
    for(int i = 0; i < NumQueues; ++i)
    {
      int d = queues[b] . template dequeueSelected<MultiProcedure>(data, procId, maxNum);
      if(d) return d;
      b = (b + 1) % NumQueues;
    }
    return 0;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = 100000)
  {
    int b = blockIdx.x % NumQueues;
    #pragma unroll
    for(int i = 0; i < NumQueues; ++i)
    {
      int d = queues[b] . template dequeueStartRead<MultiProcedure>(data,procId,maxShared);
      if(d)
      {
        if(threadIdx.x == 0)
          procId[1] |= (b << 16);
        return d;
      }
      b = (b + 1) % NumQueues;
    }
    return 0;
  }

  template<class PROCEDURE>
  __inline__ __device__ int2 reserveRead(int maxNum = -1)
  {
    int b = blockIdx.x % NumQueues;
    return queues[b]. template reserveRead<PROCEDURE>(maxNum);
  }
  template<class PROCEDURE>
  __inline__ __device__ int startRead(void*& data, int num)
  {
    int b = blockIdx.x % NumQueues;
    int id = queues[b]. template startRead<PROCEDURE>(data, num);
    return id | (b << 16);
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead(int id,  int num)
  {
    int b = id >> 16;
    return queues[b] . template finishRead<PROCEDURE>(id & 0xFFFF, num);
  }
};

template<template<class ProcedureInfo> class TInternalQueueing, int NumQueues, bool TAssertOnOverflow = true>
struct PerBlockStealingTyping 
{
  template<class ProcedureInfo>
  class Type : public  PerBlockStealing<ProcedureInfo, TInternalQueueing, NumQueues, TAssertOnOverflow> {}; 
};


template<class ProcedureInfo, template<class ProcedureInfo> class TInternalQueueing, int NumQueues, bool AssertOnOverflow = true>
class PerBlockDonating : public PerBlockQueue<ProcedureInfo, TInternalQueueing, NumQueues, AssertOnOverflow>
{
protected:
  template<class PROCEDURE>
  __device__ bool enqueueInternal(int qId, typename PROCEDURE::ExpectedData data) 
  { 
    
    #pragma unroll
    for(int i = 0; i < NumQueues; ++i, qId = (qId + 1) % NumQueues)
      if(queues[qId]. template enqueue<PROCEDURE>(data))
        return true;
    if(AssertOnOverflow)
    {
      printf("ERROR: DistributedPerBlockDonating out of elements!\n");
      Softshell::trap();
    }
    return false;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueueInternal(int qId, typename PROCEDURE::ExpectedData* data) 
  {
    #pragma unroll
    for(int i = 0; i < NumQueues; ++i, qId = (qId + 1) % NumQueues)
      if(queues[qId]. template enqueue<threads, PROCEDURE>(data))
        return true;
    if(AssertOnOverflow)
    {
      printf("ERROR: DistributedPerBlockDonating out of elements!\n");
      Softshell::trap();
    }
    return false;
  }

public:
  static std::string name()
  {
    return std::string("DistributedPerBlockDonating_") + InternalQueue::name();
  }


  template<class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  { 
    int qId = blockIdx.x % NumQueues;
    return enqueueInternal<PROCEDURE>(qId,data);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    int qId = blockIdx.x % NumQueues;
    return enqueueInternal<threads,PROCEDURE>(qId,data);
  }
};

template<template<class ProcedureInfo> class TInternalQueueing, int NumQueues, bool AssertOnOverflow = true>
struct PerBlockDonatingTyping 
{
  template<class ProcedureInfo>
  class Type : public  PerBlockDonating<ProcedureInfo, TInternalQueueing, NumQueues, AssertOnOverflow> {}; 
};

template<class ProcedureInfo, template<class ProcedureInfo> class TInternalQueueing, int NumQueues, int DonateProbability = 5, bool AssertOnOverflow = true>
class PerBlockRandomDonating : public PerBlockDonating<ProcedureInfo, TInternalQueueing, NumQueues, AssertOnOverflow>
{
public:
  static std::string name()
  {
    return std::string("DistributedPerBlockRandomDonating_") + InternalQueue::name();
  }

  template<class PROCEDURE>
  __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  { 
    int qId = blockIdx.x % NumQueues;
    if(qrandom::warp_check(DonateProbability))
      qId = qrandom::warp_rand() % NumQueues;
    return PerBlockDonating<ProcedureInfo, TInternalQueueing, NumQueues, AssertOnOverflow>:: enqueueInternal <PROCEDURE>(qId,data);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    int qId = blockIdx.x % NumQueues;
    return PerBlockDonating<ProcedureInfo, TInternalQueueing, NumQueues, AssertOnOverflow>:: enqueueInternal<threads,PROCEDURE>(qId,data);
  }
};



template<template<class ProcedureInfo> class TInternalQueueing, int NumQueues, int DonateProbability = 5, bool AssertOnOverflow = true>
struct PerBlockRandomDonatingTyping 
{
  template<class ProcedureInfo>
  class Type : public PerBlockRandomDonating<ProcedureInfo, TInternalQueueing, NumQueues, AssertOnOverflow> {}; 
};
