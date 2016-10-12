#pragma once
#include "queueInterface.cuh"
#include "procinfoTemplate.cuh"
#include "common.cuh"

template<class PROCINFO, template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
class MixedPackageQueue : public Queue<>
{
protected:
  static const uint IntElementSize = (PROCINFO::MaxDataSize + sizeof(uint) - 1)/sizeof(uint);
  typedef TInternalQueue<PROCINFO::MaxDataSize, TQueueSize, int > InternalQueue;
  InternalQueue queue;

public:
  __inline__ __device__ void init() 
  {
    queue.init();
  }
  
  template<class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    int procId = findProcId<PROCINFO, PROCEDURE>::value;
    return queue . template enqueueInitial<typename PROCEDURE::ExpectedData> ( data, procId);
  }

  template<class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {
    int procId = findProcId<PROCINFO, PROCEDURE>::value;
    return queue . template enqueue<typename PROCEDURE::ExpectedData> ( data, procId);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    int procId = findProcId<PROCINFO, PROCEDURE>::value;
    return queue . template enqueue<threads, typename PROCEDURE::ExpectedData> ( data, procId);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = -1)
  {
    return queue . dequeue (data, procId, 1);
  }

  __inline__ __device__ int size() const
  {
    return queue . size();
  }

  static std::string name()
  {
    //return "MixedPackageQueue";
    return std::string("MixedPackageQueue") + InternalQueue::name();
  }
};

template<template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
struct MixedPackageQueueTyping
{
  template<class ProcedureInfo>
  class Type : public MixedPackageQueue<ProcedureInfo, TInternalQueue, TQueueSize> {}; 
};

template<class PROCINFO, template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
class MixedPackageQueueSorted : public MixedPackageQueue<PROCINFO, TInternalQueue, TQueueSize>
{
public:
  
	static const int globalMaintainMinThreads = PROCINFO:: template Priority<0>::MinPriority != PROCINFO::template Priority<0>::MaxPriority ? 32 : 0;
	static int globalMaintainSharedMemory(int Threads) { if (PROCINFO:: template Priority<0>::MinPriority != PROCINFO:: template Priority<0>::MaxPriority) return Threads * 4 * sizeof(uint); else return 0; }

  static std::string name()
  {
    //return "MixedPackageQueue";
    return std::string("MixedPackageQueueSorted") + InternalQueue::name();
  }

  __inline__ __device__ void globalMaintain()
  { 
    queue . template sort<PROCINFO:: template Priority<0> >(blockDim.x);
  }
};

template<template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
struct MixedPackageQueueSortedTyping
{
  template<class ProcedureInfo>
  class Type : public MixedPackageQueueSorted<ProcedureInfo, TInternalQueue, TQueueSize> {}; 
};


template<class PROCINFO, template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
class MixedItemQueue : public MixedPackageQueue<PROCINFO, TInternalQueue, TQueueSize>
{


public:

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = 128*1024)
  {
    int n = min(blockDim.x, (unsigned int)(maxShared / (sizeof(int)*IntElementSize)));
    n = queue . dequeue (data, procId, n);
    if(threadIdx.x < n)
    {
      data = ((unsigned int*)data) + IntElementSize*threadIdx.x;
      procId = procId + threadIdx.x;
    }
    return n;
  }


  static std::string name()
  {
    //return "MixedPackageQueue";
    return std::string("MixedItemQueue") + InternalQueue::name();
  }
};

template<template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
struct MixedItemQueueTyping
{
  template<class ProcedureInfo>
  class Type : public MixedItemQueue<ProcedureInfo, TInternalQueue, TQueueSize> {}; 
};


template<class PROCINFO, template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
class MixedItemQueueSorted : public MixedItemQueue<PROCINFO, TInternalQueue, TQueueSize>
{
public:
  
  static const int globalMaintainMinThreads = PROCINFO::Priorities?32:0;
  static int globalMaintainSharedMemory(int Threads) { if(PROCINFO::Priorities) return Threads*4*sizeof(uint); else return 0; }

  static std::string name()
  {
    //return "MixedPackageQueue";
    return std::string("MixedItemQueueSorted") + InternalQueue::name();
  }

  __inline__ __device__ void globalMaintain()
  { 
    queue . template sort<typename PROCINFO::Priority>(blockDim.x);
  }
};

template<template <uint TElementSize, uint TQueueSize, class TAdditionalData> class TInternalQueue, uint TQueueSize>
struct MixedItemQueueSortedTyping
{
  template<class ProcedureInfo>
  class Type : public MixedItemQueueSorted<ProcedureInfo, TInternalQueue, TQueueSize> {}; 
};

