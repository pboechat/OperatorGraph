#pragma once
#include "queueExternalFetch.cuh"
#include "queueInterface.cuh"
#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"
#include "common.cuh"
#include "random.cuh"


template<class PROCINFO, class PROCEDURE,  template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PacakgeQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, bool Itemized, bool InitialQueue >
struct QueueSelector;

template<class PROCINFO, class PROCEDURE, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PacakgeQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize >
struct QueueSelector<PROCINFO, PROCEDURE, InternalPackageQueue, PacakgeQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, true, false> : public InternalItemQueue<sizeof(typename PROCEDURE::ExpectedData), ItemQueueSize, void>
{
  static const bool Itemized = true;
  typedef PROCINFO ProcInfo;
  typedef PROCEDURE Procedure;

  __inline__ __device__ void record() { }
  __inline__ __device__ void reset() { }
};
template<class PROCINFO, class PROCEDURE, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize >
struct QueueSelector<PROCINFO, PROCEDURE, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, false, false> : public InternalPackageQueue<sizeof(typename PROCEDURE::ExpectedData), PackageQueueSize, void>
{
  static const bool Itemized = false;
  typedef PROCINFO ProcInfo;
  typedef PROCEDURE Procedure;

  __inline__ __device__ void record() { }
  __inline__ __device__ void reset() { }
};


template<class PROCINFO, class PROCEDURE, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, bool TItemized >
struct QueueSelector<PROCINFO, PROCEDURE, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, TItemized, true> : public InitialDataQueue<sizeof(typename PROCEDURE::ExpectedData), InitialDataQueueSize, void>
{
  static const bool Itemized = TItemized;
  typedef PROCINFO ProcInfo;
  typedef PROCEDURE Procedure;
};
  




template<class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, bool RandomSelect = false>
class PerProcedureVersatileQueue : public ::Queue<> 
{
  
  template<class TProcedure>
  struct QueueAttachment : public QueueSelector<ProcedureInfo, TProcedure, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize,  QueueExternalFetch, 128*1024,  TProcedure::ItemInput, TProcedure::InitialProcedure >
  { };

  Attach<QueueAttachment, ProcedureInfo> queues;

  int dummy[32]; //compiler alignment mismatch hack

  template<bool MultiProcedure>
  class Visitor
  {
    uint _haveSomething;
    int*& _procId;
    void*& _data;
    const int _itemizedThreshold;
    int _maxShared;
  public:
    __inline__ __device__ Visitor(int*& procId, void*& data, int minItems, int maxShared) : 
         _haveSomething(0), _procId(procId), _data(data), _itemizedThreshold(minItems), _maxShared(maxShared)
    { }
    __inline__ __device__ uint haveSomething() const
    {
      return _haveSomething;
    }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      typedef typename TQAttachment::Procedure Procedure;
      const bool Itemized = TQAttachment::Itemized;
      
      __shared__ volatile int ssize;
      ssize = q.size();
      __syncthreads();
      int size = ssize;
      __syncthreads();
      if(size == 0) 
        return false;


      if(Itemized || MultiProcedure)
      {
        int itemThreadCount = Procedure::NumThreads > 0 ? Procedure::NumThreads : (MultiProcedure ? blockDim.x : 1);
        if(size*itemThreadCount >= _itemizedThreshold)
        {
          int nItems = Procedure::sharedMemory != 0 ? min(blockDim.x/itemThreadCount, _maxShared / ((uint)sizeof(typename Procedure::ExpectedData) + Procedure::sharedMemory)) :  min(blockDim.x/itemThreadCount, _maxShared / ((uint)sizeof(typename Procedure::ExpectedData)));
          nItems = min(nItems, getElementCount<Procedure, MultiProcedure>());
          _haveSomething = q.dequeue(_data, nItems);
          if(threadIdx.x < _haveSomething*itemThreadCount)
          {
            _data = reinterpret_cast<char*>(_data) + sizeof(typename Procedure::ExpectedData)*(threadIdx.x/itemThreadCount);
            _haveSomething *= itemThreadCount; 
            _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
          }
          return _haveSomething > 0;
        }
        return false;
      }
      else
      {
        _haveSomething = q.dequeue(_data, 1) * (Procedure::NumThreads > 0 ? Procedure::NumThreads : blockDim.x);
        _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
        return _haveSomething > 0;
      }
    }
  };


  template<bool MultiProcedure>
  class ReadVisitor
  {
    uint _haveSomething;
    int*& _procId;
    void*& _data;
    const int _itemizedThreshold;
    int _maxShared;
  public:
    __inline__ __device__ ReadVisitor(int*& procId, void*& data, int minItems, int maxShared) : 
         _haveSomething(0), _procId(procId), _data(data), _itemizedThreshold(minItems), _maxShared(maxShared)
    { }
    __inline__ __device__ uint haveSomething() const
    {
      return _haveSomething;
    }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      typedef typename TQAttachment::Procedure Procedure;
      const bool Itemized = TQAttachment::Itemized;

      __shared__ volatile int ssize;
      ssize = q.size();
      __syncthreads();
      int size = ssize;
      __syncthreads();
      if(size == 0) 
        return false;

      if(Itemized || MultiProcedure)
      {
        int itemThreadCount = Procedure::NumThreads > 0 ? Procedure::NumThreads : (MultiProcedure ? blockDim.x : 1);
        if(size*itemThreadCount >= _itemizedThreshold)
        {
          int nItems = Procedure::sharedMemory != 0 ? min(blockDim.x/itemThreadCount, _maxShared / Procedure::sharedMemory) : blockDim.x/itemThreadCount;
          nItems = min(nItems, getElementCount<Procedure, MultiProcedure>());
          _haveSomething = q.reserveRead(nItems);
          if(_haveSomething != 0)
          {
            int id = q.startRead(_data, threadIdx.x/itemThreadCount, _haveSomething);
            _haveSomething *= itemThreadCount; 
            _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
            _procId[1] = id;
            return true;
          }
        }
      }
      else
      {
        _haveSomething = q.reserveRead(1);
        if(_haveSomething != 0)
        {
          int id = q.startRead(_data, 0, _haveSomething);
          _haveSomething *= (Procedure::NumThreads > 0 ? Procedure::NumThreads : blockDim.x);
          _procId[0] = findProcId<ProcedureInfo, Procedure>::value;
          _procId[1] = id;
          return true;
        }
      }
      return false;
    }
  };

  struct NameVisitor
  {
    std::string name;
    template<class Procedure>
    bool visit()
    {
      if(name.size() > 0)
        name += ",";
      name += Procedure::name();
      return false;
    }
  };

  struct InitVisitor
  {
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q.init();
      return false;
    }
  };

  template<class TProcedure>
  struct EnqueueInitialVisitor
  {
    typename TProcedure::ExpectedData& data;
    bool res;
    __inline__ __device__ EnqueueInitialVisitor(typename TProcedure::ExpectedData& d) : data(d) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q.template enqueueInitial<typename TProcedure::ExpectedData>(data);
      return true;
    }
  };

  template<int TThreads, class TProcedure>
  struct EnqueueInitialThreadsVisitor
  {
    typename TProcedure::ExpectedData* data;
    bool res;
    __inline__ __device__ EnqueueInitialThreadsVisitor(typename TProcedure::ExpectedData* d) : data(d) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q.template enqueueInitial<TThreads, typename TProcedure::ExpectedData>(data);
      return true;
    }
  };


  template<class TProcedure>
  struct EnqueueVisitor
  {
    typename TProcedure::ExpectedData& data;
    bool res;
    __inline__ __device__ EnqueueVisitor(typename TProcedure::ExpectedData& d) : data(d) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q.template enqueue <typename TProcedure::ExpectedData>(data);
      return true;
    }
  };

  template< int Threads, class TProcedure>
  struct EnqueueThreadsVisitor
  {
    typename TProcedure::ExpectedData* data;
    bool res;
    __inline__ __device__ EnqueueThreadsVisitor(typename TProcedure::ExpectedData* d) : data(d) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q.template enqueue <Threads, typename TProcedure::ExpectedData>(data);
      return true;
    }
  };

  template<class TProcedure>
  struct ReserveSpotVisitor
  {
    typename TProcedure::ExpectedData* result;
    __inline__ __device__ ReserveSpotVisitor() : result(0) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      result = q.template reserveSpot <typename TProcedure::ExpectedData>();
      return true;
    }
  };

  template<int Threads, class TProcedure>
  struct ReserveSpotThreadsVisitor
  {
    typename TProcedure::ExpectedData* result;
    __inline__ __device__ ReserveSpotThreadsVisitor() : result(0) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      result = q.template reserveSpot <Threads, typename TProcedure::ExpectedData>();
      return true;
    }
  };

  template<class TProcedure>
  struct CompleteSpotVisitor
  {
    typename TProcedure::ExpectedData* spot;
    __inline__ __device__ CompleteSpotVisitor(typename TProcedure::ExpectedData* spot) : spot(spot) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q.template completeSpot <typename TProcedure::ExpectedData>(spot);
      return true;
    }
  };

  template<int Threads, class TProcedure>
  struct  CompleteSpotThreadsVisitor
  {
     typename TProcedure::ExpectedData* spot;
    __inline__ __device__ CompleteSpotThreadsVisitor(typename TProcedure::ExpectedData* spot) : spot(spot) { }
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q.template completeSpot <Threads, typename TProcedure::ExpectedData>(spot);
      return true;
    }
  };
   
  template<bool MultiProcedure>
  struct DequeueSelectedVisitor
  {
    void*& data;
    int maxNum;
    int res;

    __inline__ __device__ DequeueSelectedVisitor(void*& data, int maxNum) : data(data), maxNum(maxNum) { }

    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q.dequeueSelected(data, TQAttachment::ProcedureId, maxNum);
      return true;
    }
  };

  template<class TProcedure>
  struct ReserveReadVisitor
  {
    int maxNum;
    int res;

    __inline__ __device__ ReserveReadVisitor(int maxNum) : maxNum(maxNum) { }

    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q. reserveRead (maxNum);
      return true;
    }
  };

  template<class TProcedure>
  struct StartReadVisitor
  {
    void*& data;
    int num;
    int res;

    __inline__ __device__ StartReadVisitor(void*& data, int num) : data(data), num(num) { }

    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      res = q . startRead  (data, getThreadOffset<TProcedure, true>(), num);
      return true;
    }
  };

  template<class TProcedure>
  struct FinishReadVisitor
  {
    int id;
    int num;

    __inline__ __device__ FinishReadVisitor(int id, int num) : id(id), num(num) { }

    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q . finishRead (id, num);
      return true;
    }
  };


  struct NumEntriesVisitor
  {
    int* counts;
    int i;

    __inline__ __device__ NumEntriesVisitor(int* counts) : counts(counts), i(0) { }

    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      counts[i] = q.size();
      ++i;
      return false;
    }
  };


  struct RecordVisitor
  {
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q.record();
      return false;
    }
  };

  struct ResetVisitor
  {
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q.reset();
      return false;
    }
  };

  struct ClearVisitor
  {
    template<class TQAttachment>
    __inline__ __device__ bool visit(TQAttachment& q)
    {
      q.clear();
      return false;
    }
  };

  struct SizeVisitor
  {
    uint maxs;
    __inline__ __device__ SizeVisitor() : maxs(0) {}

    template<class TQAttachment>
    __inline__ __device__ bool visit(const TQAttachment& q)
    {
      maxs = max(q.size(), maxs);
      return false;
    }
    __inline__ __device__ uint result() const { return maxs;  }
  };
  

public:

  static const bool supportReuseInit = true;

  static std::string name()
  {
    //NameVisitor v;
    //ProcInfoVisitor<ProcedureInfo>::Visit<NameVisitor>(v);
    //return std::string("DistributedPerProcedure[") + v.name() + "]";
    return std::string("DistributedPerProcedure[") + InternalPackageQueue<16, PackageQueueSize, void>::name() + "," + InternalItemQueue<16, ItemQueueSize, void>::name() + "]" ;
  }

  __inline__ __device__ void init() 
  {
    InitVisitor visitor;
    queues . template VisitAll<InitVisitor>(visitor);
  }


  template<class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    EnqueueInitialVisitor<PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueInitialVisitor<PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData* data)
  {
    EnqueueInitialThreadsVisitor<threads, PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueInitialThreadsVisitor<threads, PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }

  template<class PROCEDURE>
  __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {        
    EnqueueVisitor<PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueVisitor<PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    EnqueueThreadsVisitor<threads, PROCEDURE> visitor(data);
    queues. template VisitSpecific<EnqueueThreadsVisitor<threads, PROCEDURE>, PROCEDURE>(visitor);
    return visitor.res;
  }

  template<class PROCEDURE>
  __inline__ __device__ typename PROCEDURE::ExpectedData* reserveSpot()
  {
    ReserveSpotVisitor<PROCEDURE> visitor;
    queues. template VisitSpecific<ReserveSpotVisitor<PROCEDURE>, PROCEDURE>(visitor);
    return visitor.result;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ typename PROCEDURE::ExpectedData* reserveSpot()
  {
    ReserveSpotThreadsVisitor<threads, PROCEDURE> visitor;
    queues. template VisitSpecific<ReserveSpotThreadsVisitor<threads, PROCEDURE>, PROCEDURE>(visitor);
    return visitor.result;
  }

  template<class PROCEDURE>
  __inline__ __device__ void completeSpot(typename PROCEDURE::ExpectedData* spot)
  {
    CompleteSpotVisitor<PROCEDURE> visitor(spot);
    queues. template VisitSpecific<CompleteSpotVisitor<PROCEDURE>, PROCEDURE>(visitor);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ void completeSpot(typename PROCEDURE::ExpectedData* spot)
  {
    CompleteSpotThreadsVisitor<threads, PROCEDURE> visitor(spot);
    queues. template VisitSpecific<CompleteSpotThreadsVisitor<threads, PROCEDURE>, PROCEDURE>(visitor);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = 100000)
  {     
    if(!RandomSelect)
    {
      Visitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAll<Visitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      Visitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAll<Visitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }
    else
    {
      Visitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAllRandStart<Visitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      Visitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAllRandStart<Visitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }    
    return 0;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
  {
    DequeueSelectedVisitor<MultiProcedure> visitor(data, maxNum);
    visitor.res = 0;
    queues . template VisitSpecific<DequeueSelectedVisitor<MultiProcedure> >(visitor, procId);
    return visitor.res;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = 100000)
  {
    if(!RandomSelect)
    {
      ReadVisitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAll<ReadVisitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      ReadVisitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAll<ReadVisitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }
    else
    {
      ReadVisitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
      if(queues. template VisitAllRandStart<ReadVisitor<MultiProcedure> >(visitor))
        return visitor.haveSomething();
      ReadVisitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
      if(queues. template VisitAllRandStart<ReadVisitor<MultiProcedure> >(visitor2))
        return visitor2.haveSomething();
    }
   
    return 0;
  }

  template<class PROCEDURE>
  __inline__ __device__ int reserveRead(int maxNum = -1)
  {
    if(maxNum == -1)
      maxNum = blockDim.x / (PROCEDURE::NumThreads>0 ? PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? 1 : blockDim.x));

    ReserveReadVisitor<PROCEDURE> visitor(maxNum);
    queues . template VisitSpecific<ReserveReadVisitor<PROCEDURE>,PROCEDURE >(visitor);
    return visitor.res;
  }
  template<class PROCEDURE>
  __inline__ __device__ int startRead(void*& data, int num)
  {
    StartReadVisitor<PROCEDURE> visitor(data, num);
    queues . template VisitSpecific<StartReadVisitor<PROCEDURE>,PROCEDURE >(visitor);
    return visitor.res;
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead(int id,  int num)
  {
    FinishReadVisitor<PROCEDURE> visitor(id, num);
    queues . template VisitSpecific<FinishReadVisitor<PROCEDURE>,PROCEDURE >(visitor);
  }

  __inline__ __device__ void numEntries(int* counts)
  { 
    NumEntriesVisitor visitor(counts);
    queues . template VisitAll<NumEntriesVisitor>(visitor);
  }

  __inline__ __device__ int size()
  {
    SizeVisitor visitor;
    queues . template VisitAll<SizeVisitor>(visitor);
    return visitor.result();
  }

  __inline__ __device__ void record()
  {
    RecordVisitor visitor;
    queues . template VisitAll<RecordVisitor>(visitor);
  }

  __inline__ __device__ void reset()
  {
    ResetVisitor visitor;
    queues . template VisitAll<ResetVisitor>(visitor);
  }

  __inline__ __device__ void clear()
  {
    ClearVisitor visitor;
    queues . template VisitAll<ClearVisitor>(visitor);
  }
};





template<class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint QueueSize, bool RandomSelect = false>
class PerProcedureQueue : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, QueueSize, InternalQueue, QueueSize, RandomSelect>
{
};

template<template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint PackageQueueSize,  uint ItemQueueSize, bool RandomSelect = false>
struct PerProcedureQueueDualSizeTyping 
{
  template<class ProcedureInfo>
  class Type : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, PackageQueueSize, InternalQueue, ItemQueueSize, RandomSelect> {}; 
};


template<template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint QueueSize, bool RandomSelect = false>
struct PerProcedureQueueTyping 
{
  template<class ProcedureInfo>
  class Type : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, QueueSize, InternalQueue, QueueSize, RandomSelect> {}; 
};
















////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

//template<uint TElementSize = 0, uint TQueueSize = 0, class TAdditionalData = void>
//class NoQueue : public BasicQueue<TAdditionalData>
//{
//public:
//  __inline__ __device__ void init() 
//  { }
//  
//  template<class Data>
//  __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additionalData) 
//  { return false; }
//
//  template<class Data>
//  __inline__ __device__ bool enqueue(Data data, TAdditionalData additionalData) 
//  { return false; }
//
//  template<int threads, class Data>
//  __inline__ __device__ bool enqueue(Data* data, TAdditionalData additionalData) 
//  { return false; }
//
//  __inline__ __device__ int dequeue(void* data, TAdditionalData* addtionalData, int maxnum)
//  {  return 0; }
//
//  __inline__ __device__ int size() const
//  {  return 0; }
//
//  __inline__ __device__ void record() { }
//  __inline__ __device__ void reset() { }
//
//  static std::string name()
//  {  return ""; }
//};
//template<uint TElementSize, uint TQueueSize>
//class NoQueue<TElementSize, TQueueSize, void> : public BasicQueue<void>
//{
//public:
//  __inline__ __device__ void init() 
//  { }
//  
//  template<class Data>
//  __inline__ __device__ bool enqueueInitial(Data data) 
//  { return false; }
//
//  template<class Data>
//  __inline__ __device__ bool enqueue(Data data) 
//  { return false; }
//
//  template<int threads, class Data>
//  __inline__ __device__ bool enqueue(Data* data) 
//  { return false; }
//
//  __inline__ __device__ int dequeue(void* data, int maxnum)
//  {  return 0; }
//
//  __inline__ __device__ int size() const
//  {  return 0; }
//
//  __inline__ __device__ void record() { }
//  __inline__ __device__ void reset() { }
//
//  static std::string name()
//  {  return ""; }
//};
//
//template<class PROCEDURE, class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PacakgeQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, bool Itemized, bool InitialQueue, int procIdSel >
//struct QueueSelector;
//
//template<class PROCEDURE, class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PacakgeQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, int procIdSel >
//struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, PacakgeQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, true, false, procIdSel> : public InternalItemQueue<sizeof(typename PROCEDURE::ExpectedData), ItemQueueSize, void>
//{
//  static const bool Itemized = true;
//  static const bool Empty = false;
//  typedef PROCEDURE Procedure;
//
//  __inline__ __device__ void record() { }
//  __inline__ __device__ void reset() { }
//};
//template<class PROCEDURE, class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, int procIdSel >
//struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, false, false, procIdSel> : public InternalPackageQueue<sizeof(typename PROCEDURE::ExpectedData), PackageQueueSize, void>
//{
//  static const bool Itemized = false;
//  static const bool Empty = false;
//  typedef PROCEDURE Procedure;
//
//  __inline__ __device__ void record() { }
//  __inline__ __device__ void reset() { }
//};
//template<class PROCEDURE, class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize >
//struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, true, false, -1> : public NoQueue< >
//{
//  static const bool Itemized = true;
//  static const bool Empty = true;
//  typedef PROCEDURE Procedure;
//};
//template<class PROCEDURE, class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize>
//struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, false, false, -1> : public NoQueue< >
//{
//  static const bool Itemized = false;
//  static const bool Empty = true;
//  typedef PROCEDURE Procedure;
//};
//template<class PROCEDURE, class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue, uint InitialDataQueueSize, bool TItemized, int procIdSel >
//struct QueueSelector<PROCEDURE, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, TItemized, true, procIdSel> : public InitialDataQueue<sizeof(typename PROCEDURE::ExpectedData), InitialDataQueueSize, void>
//{
//  static const bool Itemized = TItemized;
//  static const bool Empty = false;
//  typedef PROCEDURE Procedure;
//};
//  
//
//
//
//template<class NEWPROCEDURE, class PROCEDUREPRESENT, class TheQueue>
//class QueueMatchSelector
//{
//public:
//  __inline__ __device__ static bool enqueueInitial(TheQueue& q, typename NEWPROCEDURE::ExpectedData data) 
//  {  return false;  }
//
//  __inline__ __device__ static bool enqueue(TheQueue& q, typename NEWPROCEDURE::ExpectedData data) 
//  { return false;    }
//
//  template<int threads>
//  __inline__ __device__ static bool enqueue(TheQueue& q, typename NEWPROCEDURE::ExpectedData* data) 
//  { return false;    }
//
//
//  __inline__ __device__ static int dequeue(TheQueue& q, typename NEWPROCEDURE::ExpectedData* data, int* procId, int num)
//  { return 0; }
//    
//  __inline__ __device__ static int size()
//  { return 0; }
//
//  __inline__ __device__ static int reserveRead(TheQueue& q, int maxnum, bool only_read_all)
//  { return 0; }
//  __inline__ __device__ static int startRead(TheQueue& q, void*& data, int pos, int num)
//  { return 0; }
//  __inline__ __device__ static void finishRead(TheQueue& q, int id, int num)
//  { }
//};
//
//template< class PROCEDUREMATCH, class TheQueue>
//class QueueMatchSelector<PROCEDUREMATCH, PROCEDUREMATCH, TheQueue>
//{
//public:
//  __inline__ __device__ static bool enqueueInitial(TheQueue& q, typename PROCEDUREMATCH::ExpectedData data) 
//  {
//    return q . template enqueueInitial<PROCEDUREMATCH::ExpectedData>(data);
//  }
//  __inline__ __device__ static bool enqueue(TheQueue& q, typename PROCEDUREMATCH::ExpectedData data) 
//  { 
//    return q . template enqueue<PROCEDUREMATCH::ExpectedData>(data);
//  }
//  template<int threads>
//  __inline__ __device__ static bool enqueue(TheQueue& q, typename PROCEDUREMATCH::ExpectedData* data) 
//  {
//    return q . template enqueue<threads, PROCEDUREMATCH::ExpectedData>(data); 
//  }
//
//  __inline__ __device__ static int dequeue(TheQueue& q, void* data, int num)
//  {  
//    return q . dequeue(data, num);
//  }
//
//  __inline__ __device__ static int size(const TheQueue& q)
//  {  
//    return q . size();
//  }
//
//  __inline__ __device__ static int reserveRead(TheQueue& q, int maxnum, bool only_read_all)
//  { 
//    return q . reserveRead(maxnum, only_read_all); 
//  }
//  __inline__ __device__ static int startRead(TheQueue& q, void*& data, int pos, int num)
//  { 
//    return q . startRead(data, pos, num); 
//  }
//  __inline__ __device__ static void finishRead(TheQueue& q, int id, int num)
//  { 
//    q . finishRead(id, num); 
//  }
//};
//
//template<class Q, bool Empty, class Visitor>
//class QueueVisitor
//{
//public:
//  __inline__ __device__ 
//  static bool visit(Q& q, Visitor& visitor)
//  {
//    return visitor. template visit<Q, typename Q::Procedure, Q::Itemized > (q);
//  }
//};
//template<class Q, class Visitor>
//class QueueVisitor<Q, true, Visitor>
//{
//public:
//  __inline__ __device__ 
//  static bool visit(Q& q, Visitor& visitor)
//  {
//    return false;
//  }
//};
//
//
//
//template<class ProcedureInfo, template <uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InitialDataQueue = QueueExternalFetch, uint InitialDataQueueSize = 128*1024>
//class MultiQueue 
//{
//  typedef QueueSelector<typename ProcedureInfo::Procedure0,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure0::ItemInput,  ProcedureInfo::Procedure0::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure0::ProcedureId>::Id>  Q0_t;  Q0_t Q0;
//  typedef QueueSelector<typename ProcedureInfo::Procedure1,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure1::ItemInput,  ProcedureInfo::Procedure1::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure1::ProcedureId>::Id>  Q1_t;  Q1_t Q1;
//  typedef QueueSelector<typename ProcedureInfo::Procedure2,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure2::ItemInput,  ProcedureInfo::Procedure2::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure2::ProcedureId>::Id>  Q2_t;  Q2_t Q2;
//  typedef QueueSelector<typename ProcedureInfo::Procedure3,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure3::ItemInput,  ProcedureInfo::Procedure3::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure3::ProcedureId>::Id>  Q3_t;  Q3_t Q3;
//  typedef QueueSelector<typename ProcedureInfo::Procedure4,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure4::ItemInput,  ProcedureInfo::Procedure4::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure4::ProcedureId>::Id>  Q4_t;  Q4_t Q4;
//  typedef QueueSelector<typename ProcedureInfo::Procedure5,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure5::ItemInput,  ProcedureInfo::Procedure5::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure5::ProcedureId>::Id>  Q5_t;  Q5_t Q5;
//  typedef QueueSelector<typename ProcedureInfo::Procedure6,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure6::ItemInput,  ProcedureInfo::Procedure6::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure6::ProcedureId>::Id>  Q6_t;  Q6_t Q6;
//  typedef QueueSelector<typename ProcedureInfo::Procedure7,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure7::ItemInput,  ProcedureInfo::Procedure7::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure7::ProcedureId>::Id>  Q7_t;  Q7_t Q7;
//  typedef QueueSelector<typename ProcedureInfo::Procedure8,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure8::ItemInput,  ProcedureInfo::Procedure8::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure8::ProcedureId>::Id>  Q8_t;  Q8_t Q8;
//  typedef QueueSelector<typename ProcedureInfo::Procedure9,  ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure9::ItemInput,  ProcedureInfo::Procedure9::InitialProcedure,  ProcIdSelector<ProcedureInfo::Procedure9::ProcedureId>::Id>  Q9_t;  Q9_t Q9;
//  typedef QueueSelector<typename ProcedureInfo::Procedure10, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure10::ItemInput, ProcedureInfo::Procedure10::InitialProcedure, ProcIdSelector<ProcedureInfo::Procedure10::ProcedureId>::Id> Q10_t; Q10_t Q10;
//  typedef QueueSelector<typename ProcedureInfo::Procedure11, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure11::ItemInput, ProcedureInfo::Procedure11::InitialProcedure, ProcIdSelector<ProcedureInfo::Procedure11::ProcedureId>::Id> Q11_t; Q11_t Q11;
//  typedef QueueSelector<typename ProcedureInfo::Procedure12, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure12::ItemInput, ProcedureInfo::Procedure12::InitialProcedure, ProcIdSelector<ProcedureInfo::Procedure12::ProcedureId>::Id> Q12_t; Q12_t Q12;
//  typedef QueueSelector<typename ProcedureInfo::Procedure13, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure13::ItemInput, ProcedureInfo::Procedure13::InitialProcedure, ProcIdSelector<ProcedureInfo::Procedure13::ProcedureId>::Id> Q13_t; Q13_t Q13;
//  typedef QueueSelector<typename ProcedureInfo::Procedure14, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure14::ItemInput, ProcedureInfo::Procedure14::InitialProcedure, ProcIdSelector<ProcedureInfo::Procedure14::ProcedureId>::Id> Q14_t; Q14_t Q14;
//  typedef QueueSelector<typename ProcedureInfo::Procedure15, ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize, InitialDataQueue, InitialDataQueueSize, ProcedureInfo::Procedure15::ItemInput, ProcedureInfo::Procedure15::InitialProcedure, ProcIdSelector<ProcedureInfo::Procedure15::ProcedureId>::Id> Q15_t; Q15_t Q15;
//
//  static const int NumQueues = ProcedureInfo::NumProcedures;
//
//public:
//  __inline__ __device__ void init() 
//  {
//    Q0.init();
//    Q1.init();
//    Q2.init();
//    Q3.init();
//    Q4.init();
//    Q5.init();
//    Q6.init();
//    Q7.init();
//    Q8.init();
//    Q9.init();
//    Q10.init();
//    Q11.init();
//    Q12.init();
//    Q13.init();
//    Q14.init();
//    Q15.init();
//  }
//
//    template<class PROCEDURE>
//  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
//  {
//    return
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::enqueueInitial(Q0, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::enqueueInitial(Q1, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::enqueueInitial(Q2, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::enqueueInitial(Q3, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::enqueueInitial(Q4, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::enqueueInitial(Q5, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::enqueueInitial(Q6, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::enqueueInitial(Q7, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::enqueueInitial(Q8, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::enqueueInitial(Q9, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::enqueueInitial(Q10, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::enqueueInitial(Q11, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::enqueueInitial(Q12, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::enqueueInitial(Q13, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::enqueueInitial(Q14, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::enqueueInitial(Q15, data);
//  }
//
//  template<class PROCEDURE>
//  __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
//  {     
//    return
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::enqueue(Q0, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::enqueue(Q1, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::enqueue(Q2, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::enqueue(Q3, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::enqueue(Q4, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::enqueue(Q5, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::enqueue(Q6, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::enqueue(Q7, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::enqueue(Q8, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::enqueue(Q9, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::enqueue(Q10, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::enqueue(Q11, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::enqueue(Q12, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::enqueue(Q13, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::enqueue(Q14, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::enqueue(Q15, data);
//  }
//
//  template<int threads, class PROCEDURE>
//  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
//  {
//    return
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>:: template enqueue<threads>(Q0, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>:: template enqueue<threads>(Q1, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>:: template enqueue<threads>(Q2, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>:: template enqueue<threads>(Q3, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>:: template enqueue<threads>(Q4, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>:: template enqueue<threads>(Q5, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>:: template enqueue<threads>(Q6, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>:: template enqueue<threads>(Q7, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>:: template enqueue<threads>(Q8, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>:: template enqueue<threads>(Q9, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>:: template enqueue<threads>(Q10, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>:: template enqueue<threads>(Q11, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>:: template enqueue<threads>(Q12, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>:: template enqueue<threads>(Q13, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>:: template enqueue<threads>(Q14, data) || 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>:: template enqueue<threads>(Q15, data);
//  }
//
//  template<class PROCEDURE>
//  __inline__ __device__ int dequeue(void* data, int* procId, int num)
//  {
//    return 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::dequeue(Q0, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::dequeue(Q1, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::dequeue(Q2, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::dequeue(Q3, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::dequeue(Q4, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::dequeue(Q5, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::dequeue(Q6, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::dequeue(Q7, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::dequeue(Q8, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::dequeue(Q9, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::dequeue(Q10, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::dequeue(Q11, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::dequeue(Q12, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::dequeue(Q13, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::dequeue(Q14, data, procId, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::dequeue(Q15, data, procId, num);
//  }
//
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int num)
//  {
//  #define DS_CASE(ProcNum) \
//    /*case ProcedureInfo:: Procedure ## ProcNum ::ProcedureId:*/ \
//    else if(ProcedureInfo:: Procedure ## ProcNum ::ProcedureId == procId) \
//    { \
//      nThreads = ProcedureInfo:: Procedure ## ProcNum ::NumThreads>0 ? ProcedureInfo:: Procedure ## ProcNum ::NumThreads : (ProcedureInfo:: Procedure ## ProcNum ::ItemInput ? 1 : blockDim.x); \
//      num =  Q ## ProcNum .dequeue(data, num >= 0 ? num : blockDim.x/nThreads); \
//      data =  reinterpret_cast<typename ProcedureInfo:: Procedure ## ProcNum ::ExpectedData*>(data) + threadIdx.x / nThreads; \
//      __syncthreads(); \
//      return num*nThreads; \
//    }
//
//
//    int nThreads;
//    //switch(procId)
//    //{
//    if(0 == 1) { }
//      DS_CASE(0)  DS_CASE(1)  DS_CASE(2)  DS_CASE(3)
//      DS_CASE(4)  DS_CASE(5)  DS_CASE(6)  DS_CASE(7)
//      DS_CASE(8)  DS_CASE(9)  DS_CASE(10) DS_CASE(11)
//      DS_CASE(12) DS_CASE(13) DS_CASE(14) DS_CASE(15) 
//    //}
//#undef DS_CASE
//    return 0;
//  }
//
//  
//  template<class PROCEDURE>
//  __inline__ __device__ int reserveRead(int maxNum)
//  {
//    return 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::reserveRead(Q0, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::reserveRead(Q1, maxNum, false) +
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::reserveRead(Q2, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::reserveRead(Q3, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::reserveRead(Q4, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::reserveRead(Q5, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::reserveRead(Q6, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::reserveRead(Q7, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::reserveRead(Q8, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::reserveRead(Q9, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::reserveRead(Q10, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::reserveRead(Q11, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::reserveRead(Q12, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::reserveRead(Q13, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::reserveRead(Q14, maxNum, false) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::reserveRead(Q15, maxNum, false);
//  }
//  template<class PROCEDURE>
//  __inline__ __device__ int startRead(void*& data, int pos, int num)
//  {
//    return 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::startRead(Q0, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::startRead(Q1, data, pos, num) +
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::startRead(Q2, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::startRead(Q3, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::startRead(Q4, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::startRead(Q5, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::startRead(Q6, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::startRead(Q7, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::startRead(Q8, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::startRead(Q9, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::startRead(Q10, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::startRead(Q11, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::startRead(Q12, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::startRead(Q13, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::startRead(Q14, data, pos, num) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::startRead(Q15, data, pos, num);
//  }
//  template<class PROCEDURE>
//  __inline__ __device__ void finishRead(int id,  int num)
//  {
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::finishRead(Q0, id, num);
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::finishRead(Q1, id, num);
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::finishRead(Q2, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::finishRead(Q3, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::finishRead(Q4, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::finishRead(Q5, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::finishRead(Q6, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::finishRead(Q7, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::finishRead(Q8, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::finishRead(Q9, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::finishRead(Q10, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::finishRead(Q11, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::finishRead(Q12, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::finishRead(Q13, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::finishRead(Q14, id, num); 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::finishRead(Q15, id, num);
//  }
//
// 
//
//  template<class PROCEDURE>
//  __inline__ __device__ int size() const
//  {
//    return 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure0, Q0_t>::size(Q0) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure1, Q1_t>::size(Q1) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure2, Q2_t>::size(Q2) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure3, Q3_t>::size(Q3) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure4, Q4_t>::size(Q4) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure5, Q5_t>::size(Q5) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure6, Q6_t>::size(Q6) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure7, Q7_t>::size(Q7) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure8, Q8_t>::size(Q8) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure9, Q9_t>::size(Q9) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure10, Q10_t>::size(Q10) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure11, Q11_t>::size(Q11) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure12, Q12_t>::size(Q12) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure13, Q13_t>::size(Q13) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure14, Q14_t>::size(Q14) + 
//    QueueMatchSelector<PROCEDURE, typename ProcedureInfo::Procedure15, Q15_t>::size(Q15);
//  }
//
//  static std::string name()
//  {
//
//    return 
//    Q0_t::name() + 
//    Q1_t::name() + 
//    Q2_t::name() + 
//    Q3_t::name() + 
//    Q4_t::name() + 
//    Q5_t::name() + 
//    Q6_t::name() + 
//    Q7_t::name() + 
//    Q8_t::name() + 
//    Q9_t::name() + 
//    Q10_t::name() + 
//    Q11_t::name() + 
//    Q12_t::name() + 
//    Q13_t::name() + 
//    Q14_t::name() + 
//    Q15_t::name();
//  }
//
//  template<class Visitor>
//  __inline__ __device__ bool visitAll(Visitor& visit)
//  {
//    if(QueueVisitor<Q0_t, Q0_t::Empty, Visitor>::visit(Q0, visit)) return true;
//    if(QueueVisitor<Q1_t, Q1_t::Empty, Visitor>::visit(Q1, visit)) return true;
//    if(QueueVisitor<Q2_t, Q2_t::Empty, Visitor>::visit(Q2, visit)) return true;
//    if(QueueVisitor<Q3_t, Q3_t::Empty, Visitor>::visit(Q3, visit)) return true;
//    if(QueueVisitor<Q4_t, Q4_t::Empty, Visitor>::visit(Q4, visit)) return true;
//    if(QueueVisitor<Q5_t, Q5_t::Empty, Visitor>::visit(Q5, visit)) return true;
//    if(QueueVisitor<Q6_t, Q6_t::Empty, Visitor>::visit(Q6, visit)) return true;
//    if(QueueVisitor<Q7_t, Q7_t::Empty, Visitor>::visit(Q7, visit)) return true;
//    if(QueueVisitor<Q8_t, Q8_t::Empty, Visitor>::visit(Q8, visit)) return true;
//    if(QueueVisitor<Q9_t, Q9_t::Empty, Visitor>::visit(Q9, visit)) return true;
//    if(QueueVisitor<Q10_t, Q10_t::Empty, Visitor>::visit(Q10, visit)) return true;
//    if(QueueVisitor<Q11_t, Q11_t::Empty, Visitor>::visit(Q11, visit)) return true;
//    if(QueueVisitor<Q12_t, Q12_t::Empty, Visitor>::visit(Q12, visit)) return true;
//    if(QueueVisitor<Q13_t, Q13_t::Empty, Visitor>::visit(Q13, visit)) return true;
//    if(QueueVisitor<Q14_t, Q14_t::Empty, Visitor>::visit(Q14, visit)) return true;
//    if(QueueVisitor<Q15_t, Q15_t::Empty, Visitor>::visit(Q15, visit)) return true;
//    return false;
//  }
//
//  
//  template<class Visitor>
//  __inline__ __device__ bool blockVisitRandomStart(Visitor& visit)
//  {
//    __shared__ uint startVisitAt;
//    __syncthreads();
//    startVisitAt = random::rand() % NumQueues;
//    __syncthreads();
//    switch(startVisitAt)
//    {
//    case 0:
//      if(QueueVisitor<Q0_t, Q0_t::Empty, Visitor>::visit(Q0, visit)) return true;
//    case 1:
//      if(QueueVisitor<Q1_t, Q1_t::Empty, Visitor>::visit(Q1, visit)) return true;
//    case 2:
//      if(QueueVisitor<Q2_t, Q2_t::Empty, Visitor>::visit(Q2, visit)) return true;
//    case 3:
//      if(QueueVisitor<Q3_t, Q3_t::Empty, Visitor>::visit(Q3, visit)) return true;
//    case 4:
//      if(QueueVisitor<Q4_t, Q4_t::Empty, Visitor>::visit(Q4, visit)) return true;
//    case 5:
//      if(QueueVisitor<Q5_t, Q5_t::Empty, Visitor>::visit(Q5, visit)) return true;
//    case 6:
//      if(QueueVisitor<Q6_t, Q6_t::Empty, Visitor>::visit(Q6, visit)) return true;
//    case 7:
//      if(QueueVisitor<Q7_t, Q7_t::Empty, Visitor>::visit(Q7, visit)) return true;
//    case 8:
//      if(QueueVisitor<Q8_t, Q8_t::Empty, Visitor>::visit(Q8, visit)) return true;
//    case 9:
//      if(QueueVisitor<Q9_t, Q9_t::Empty, Visitor>::visit(Q9, visit)) return true;
//    case 10:
//      if(QueueVisitor<Q10_t, Q10_t::Empty, Visitor>::visit(Q10, visit)) return true;
//    case 11:
//      if(QueueVisitor<Q11_t, Q11_t::Empty, Visitor>::visit(Q11, visit)) return true;
//    case 12:
//      if(QueueVisitor<Q12_t, Q12_t::Empty, Visitor>::visit(Q12, visit)) return true;
//    case 13:
//      if(QueueVisitor<Q13_t, Q13_t::Empty, Visitor>::visit(Q13, visit)) return true;
//    case 14:
//      if(QueueVisitor<Q14_t, Q14_t::Empty, Visitor>::visit(Q14, visit)) return true;
//    case 15:
//      if(QueueVisitor<Q15_t, Q15_t::Empty, Visitor>::visit(Q15, visit)) return true;
//
//      if(startVisitAt == 0) return false;
//      if(QueueVisitor<Q0_t, Q0_t::Empty, Visitor>::visit(Q0, visit)) return true;
//      if(startVisitAt == 1) return false;
//      if(QueueVisitor<Q1_t, Q1_t::Empty, Visitor>::visit(Q1, visit)) return true;
//      if(startVisitAt == 2) return false;
//      if(QueueVisitor<Q2_t, Q2_t::Empty, Visitor>::visit(Q2, visit)) return true;
//      if(startVisitAt == 3) return false;
//      if(QueueVisitor<Q3_t, Q3_t::Empty, Visitor>::visit(Q3, visit)) return true;
//      if(startVisitAt == 4) return false;
//      if(QueueVisitor<Q4_t, Q4_t::Empty, Visitor>::visit(Q4, visit)) return true;
//      if(startVisitAt == 5) return false;
//      if(QueueVisitor<Q5_t, Q5_t::Empty, Visitor>::visit(Q5, visit)) return true;
//      if(startVisitAt == 6) return false;
//      if(QueueVisitor<Q6_t, Q6_t::Empty, Visitor>::visit(Q6, visit)) return true;
//      if(startVisitAt == 7) return false;
//      if(QueueVisitor<Q7_t, Q7_t::Empty, Visitor>::visit(Q7, visit)) return true;
//      if(startVisitAt == 8) return false;
//      if(QueueVisitor<Q8_t, Q8_t::Empty, Visitor>::visit(Q8, visit)) return true;
//      if(startVisitAt == 9) return false;
//      if(QueueVisitor<Q9_t, Q9_t::Empty, Visitor>::visit(Q9, visit)) return true;
//      if(startVisitAt == 10) return false;
//      if(QueueVisitor<Q10_t, Q10_t::Empty, Visitor>::visit(Q10, visit)) return true;
//      if(startVisitAt == 11) return false;
//      if(QueueVisitor<Q11_t, Q11_t::Empty, Visitor>::visit(Q11, visit)) return true;
//      if(startVisitAt == 12) return false;
//      if(QueueVisitor<Q12_t, Q12_t::Empty, Visitor>::visit(Q12, visit)) return true;
//      if(startVisitAt == 13) return false;
//      if(QueueVisitor<Q13_t, Q13_t::Empty, Visitor>::visit(Q13, visit)) return true;
//      if(startVisitAt == 14) return false;
//      if(QueueVisitor<Q14_t, Q14_t::Empty, Visitor>::visit(Q14, visit)) return true;
//      if(startVisitAt == 15) return false;
//      if(QueueVisitor<Q15_t, Q15_t::Empty, Visitor>::visit(Q15, visit)) return true;
//    }
//
//    return false;
//
//  }
// 
//
//  __inline__ __device__ void numEntries(int* counts)
//  {
//    if(threadIdx.x == 0)
//    {
//      if(!Q0_t::Empty)  counts[0]  = Q0.size();
//      if(!Q1_t::Empty)  counts[1]  = Q1.size();
//      if(!Q2_t::Empty)  counts[2]  = Q2.size();
//      if(!Q3_t::Empty)  counts[3]  = Q3.size();
//      if(!Q4_t::Empty)  counts[4]  = Q4.size();
//      if(!Q5_t::Empty)  counts[5]  = Q5.size();
//      if(!Q6_t::Empty)  counts[6]  = Q6.size();
//      if(!Q7_t::Empty)  counts[7]  = Q7.size();
//      if(!Q8_t::Empty)  counts[8]  = Q8.size();
//      if(!Q9_t::Empty)  counts[9]  = Q9.size();
//      if(!Q10_t::Empty) counts[10] = Q10.size();
//      if(!Q11_t::Empty) counts[11] = Q11.size();
//      if(!Q12_t::Empty) counts[12] = Q12.size();
//      if(!Q13_t::Empty) counts[13] = Q13.size();
//      if(!Q14_t::Empty) counts[14] = Q14.size();
//      if(!Q15_t::Empty) counts[15] = Q15.size();
//    }
//    
//  }
//
//  __inline__ __device__ void record()
//  {
//    Q0.record();
//    Q1.record();
//    Q2.record();
//    Q3.record();
//    Q4.record();
//    Q5.record();
//    Q6.record();
//    Q7.record();
//    Q8.record();
//    Q9.record();
//    Q10.record();
//    Q11.record();
//    Q12.record();
//    Q13.record();
//    Q14.record();
//    Q15.record();
//  }
//
//  __inline__ __device__ void reset()
//  {
//    Q0.reset();
//    Q1.reset();
//    Q2.reset();
//    Q3.reset();
//    Q4.reset();
//    Q5.reset();
//    Q6.reset();
//    Q7.reset();
//    Q8.reset();
//    Q9.reset();
//    Q10.reset();
//    Q11.reset();
//    Q12.reset();
//    Q13.reset();
//    Q14.reset();
//    Q15.reset();
//  }
//};
//
//
//template<class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalPackageQueue, uint PackageQueueSize, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalItemQueue, uint ItemQueueSize, bool RandomSelect = false>
//class PerProcedureVersatileQueue : public ::Queue<> 
//{
//  typedef MultiQueue<ProcedureInfo, InternalPackageQueue, PackageQueueSize, InternalItemQueue, ItemQueueSize>  MyMultiQueue;
//  MyMultiQueue queues;
//
//  int dummy[32]; //compiler alignment mismatch hack
//
//  template<bool MultiProcedure>
//  class Visitor
//  {
//    uint _haveSomething;
//    int*& _procId;
//    void*& _data;
//    const int _itemizedThreshold;
//    int _maxShared;
//  public:
//    __inline__ __device__ Visitor(int*& procId, void*& data, int minItems, int maxShared) : 
//         _haveSomething(0), _procId(procId), _data(data), _itemizedThreshold(minItems), _maxShared(maxShared)
//    { }
//    __inline__ __device__ uint haveSomething() const
//    {
//      return _haveSomething;
//    }
//    template<class Queue, class Procedure, bool Itemized>
//    __inline__ __device__ bool visit(Queue& q)
//    {
//      __shared__ volatile int ssize;
//      ssize = q.size();
//      __syncthreads();
//      int size = ssize;
//      __syncthreads();
//      if(size == 0) 
//        return false;
//
//
//      if(Itemized || MultiProcedure)
//      {
//        int itemThreadCount = Procedure::NumThreads > 0 ? Procedure::NumThreads : (MultiProcedure ? blockDim.x : 1);
//        if(size*itemThreadCount >= _itemizedThreshold)
//        {
//          int nItems = Procedure::sharedMemory != 0 ? min(blockDim.x/itemThreadCount, _maxShared / ((uint)sizeof(Procedure::ExpectedData) + Procedure::sharedMemory)) :  min(blockDim.x/itemThreadCount, _maxShared / ((uint)sizeof(Procedure::ExpectedData)));
//          nItems = min(nItems, getElementCount<Procedure, MultiProcedure>());
//          _haveSomething = q.dequeue(_data, nItems);
//          if(threadIdx.x < _haveSomething*itemThreadCount)
//          {
//            _data = reinterpret_cast<char*>(_data) + sizeof(Procedure::ExpectedData)*(threadIdx.x/itemThreadCount);
//            _haveSomething *= itemThreadCount; 
//            _procId[0] = Procedure::ProcedureId;
//          }
//          return _haveSomething > 0;
//        }
//        return false;
//      }
//      else
//      {
//        _haveSomething = q.dequeue(_data, 1) * (Procedure::NumThreads > 0 ? Procedure::NumThreads : blockDim.x);
//        _procId[0] = Procedure::ProcedureId;
//        return _haveSomething > 0;
//      }
//    }
//  };
//
//
//  template<bool MultiProcedure>
//  class ReadVisitor
//  {
//    uint _haveSomething;
//    int*& _procId;
//    void*& _data;
//    const int _itemizedThreshold;
//    int _maxShared;
//  public:
//    __inline__ __device__ ReadVisitor(int*& procId, void*& data, int minItems, int maxShared) : 
//         _haveSomething(0), _procId(procId), _data(data), _itemizedThreshold(minItems), _maxShared(maxShared)
//    { }
//    __inline__ __device__ uint haveSomething() const
//    {
//      return _haveSomething;
//    }
//    template<class Queue, class Procedure, bool Itemized>
//    __inline__ __device__ bool visit(Queue& q)
//    {
//      __shared__ volatile int ssize;
//      ssize = q.size();
//      __syncthreads();
//      int size = ssize;
//      __syncthreads();
//      if(size == 0) 
//        return false;
//
//      if(Itemized || MultiProcedure)
//      {
//        int itemThreadCount = Procedure::NumThreads > 0 ? Procedure::NumThreads : (MultiProcedure ? blockDim.x : 1);
//        if(size*itemThreadCount >= _itemizedThreshold)
//        {
//          int nItems = Procedure::sharedMemory != 0 ? min(blockDim.x/itemThreadCount, _maxShared / Procedure::sharedMemory) : blockDim.x/itemThreadCount;
//          nItems = min(nItems, getElementCount<Procedure, MultiProcedure>());
//          _haveSomething = q.reserveRead(nItems);
//          if(_haveSomething != 0)
//          {
//            int id = q.startRead(_data, threadIdx.x/itemThreadCount, _haveSomething);
//            _haveSomething *= itemThreadCount; 
//            _procId[0] = Procedure::ProcedureId;
//            _procId[1] = id;
//            return true;
//          }
//        }
//      }
//      else
//      {
//        _haveSomething = q.reserveRead(1);
//        if(_haveSomething != 0)
//        {
//          int id = q.startRead(_data, 0, _haveSomething);
//          _haveSomething *= (Procedure::NumThreads > 0 ? Procedure::NumThreads : blockDim.x);
//          _procId[0] = Procedure::ProcedureId;
//          _procId[1] = id;
//          return true;
//        }
//      }
//      return false;
//    }
//  };
//
//public:
//
//  static const bool supportReuseInit = true;
//
//  static std::string name()
//  {
//    return std::string("DistributedPerProcedure[") + InternalPackageQueue<16, PackageQueueSize, void>::name() + "," + InternalItemQueue<16, ItemQueueSize, void>::name() + "]" ;
//    //std::stringstream sstr;
//    //sstr << "DistributedPerProcedure[" << MyMultiQueue::name() << "]";
//    //return sstr.str();
//  }
//
//  __inline__ __device__ void init() 
//  {
//    queues.init();
//  }
//
//
//  template<class PROCEDURE>
//  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
//  {
//    return queues. template enqueueInitial<PROCEDURE>(data);
//  }
//
//  template<class PROCEDURE>
//  __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
//  {        
//    return queues. template enqueue<PROCEDURE>(data);
//  }
//
//  template<int threads, class PROCEDURE>
//  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
//  {
//    return queues . template enqueue<threads, PROCEDURE> ( data );
//  }
//
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = 100000)
//  {     
//    if(!RandomSelect)
//    {
//      Visitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
//      if(queues. template visitAll<Visitor<MultiProcedure> >(visitor))
//        return visitor.haveSomething();
//      Visitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
//      if(queues. template visitAll<Visitor<MultiProcedure> >(visitor2))
//        return visitor2.haveSomething();
//    }
//    else
//    {
//      Visitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
//      if(queues. template blockVisitRandomStart<Visitor<MultiProcedure> >(visitor))
//        return visitor.haveSomething();
//      Visitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
//      if(queues. template blockVisitRandomStart<Visitor<MultiProcedure> >(visitor2))
//        return visitor2.haveSomething();
//    }    
//    return 0;
//  }
//
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
//  {
//    return queues.dequeueSelected(data, procId, maxNum);
//  }
//
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = 100000)
//  {
//    if(!RandomSelect)
//    {
//      ReadVisitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
//      if(queues. template visitAll<ReadVisitor<MultiProcedure> >(visitor))
//        return visitor.haveSomething();
//      ReadVisitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
//      if(queues. template visitAll<ReadVisitor<MultiProcedure> >(visitor2))
//        return visitor2.haveSomething();
//    }
//    else
//    {
//      ReadVisitor<MultiProcedure> visitor(procId, data, blockDim.x, maxShared);
//      if(queues. template blockVisitRandomStart<ReadVisitor<MultiProcedure> >(visitor))
//        return visitor.haveSomething();
//      ReadVisitor<MultiProcedure> visitor2(procId, data, 0, maxShared);
//      if(queues. template blockVisitRandomStart<ReadVisitor<MultiProcedure> >(visitor2))
//        return visitor2.haveSomething();
//    }
//   
//    return 0;
//  }
//
//  template<class PROCEDURE>
//  __inline__ __device__ int reserveRead(int maxNum = -1)
//  {
//    if(maxNum == -1)
//      maxNum = blockDim.x / (PROCEDURE::NumThreads>0 ? PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? 1 : blockDim.x));
//    return queues . template reserveRead <PROCEDURE> (maxNum);
//  }
//  template<class PROCEDURE>
//  __inline__ __device__ int startRead(void*& data, int num)
//  {
//    return queues . template startRead <PROCEDURE> (data, getThreadOffset<PROCEDURE, true>(), num);
//  }
//  template<class PROCEDURE>
//  __inline__ __device__ void finishRead(int id,  int num)
//  {
//    return queues . template finishRead <PROCEDURE> (id, num);
//  }
//
//  __inline__ __device__ void numEntries(int* counts)
//  { 
//    queues . numEntries(counts);
//  }
//
//  __inline__ __device__ void record()
//  {
//    queues.record();
//  }
//
//  __inline__ __device__ void reset()
//  {
//    queues.reset();
//  }
//};
//
//template<class ProcedureInfo, template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint QueueSize, bool RandomSelect = false>
//class PerProcedureQueue : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, QueueSize, InternalQueue, QueueSize, RandomSelect>
//{
//};
//
//template<template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint PackageQueueSize,  uint ItemQueueSize, bool RandomSelect = false>
//struct PerProcedureQueueDualSizeTyping 
//{
//  template<class ProcedureInfo>
//  class Type : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, PackageQueueSize, InternalQueue, ItemQueueSize, RandomSelect> {}; 
//};
//
//
//template<template<uint TElementSize, uint TQueueSize, class TAdditionalData> class InternalQueue, uint QueueSize, bool RandomSelect = false>
//struct PerProcedureQueueTyping 
//{
//  template<class ProcedureInfo>
//  class Type : public PerProcedureVersatileQueue<ProcedureInfo, InternalQueue, QueueSize, InternalQueue, QueueSize, RandomSelect> {}; 
//};
