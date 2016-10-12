#pragma once
#include <string>

template<bool TWarnings = true>
class Queue
{
public:
  static const bool needTripleCall = false;
  static const bool supportReuseInit = false;
  static const int globalMaintainMinThreads = 0;
  static int globalMaintainSharedMemory(int Threads) { return 0; }
  static const int requiredShared = 0;

  __inline__ __device__ void init() 
  {
    if(TWarnings) printf("Warning: Queue does not implement init\n");
  }
  
  template<class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    if(TWarnings) printf("Error: Queue does not implement enqueueInitial\n");
    return false;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData* data) 
  {
    if(TWarnings) printf("Error: Queue does not implement enqueueInitial<int,Proc>\n");
    return false;
  }

  template<class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {
    if(TWarnings) printf("Error: Queue does not implement enqueue\n");
    return false;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    if(TWarnings) printf("Error: Queue does not implement enqueue<Threads>\n");
    return false;
  }

  template<class PROCEDURE>
  __inline__ __device__ typename PROCEDURE::ExpectedData* reserveSpot()
  {
    if(TWarnings) printf("Error: Queue does not implement reserveSpot\n");
    return 0;
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ typename PROCEDURE::ExpectedData* reserveSpot()
  {
    if(TWarnings) printf("Error: Queue does not implement reserveSpot<Threads>\n");
    return 0;
  }

  template<class PROCEDURE>
  __inline__ __device__ void completeSpot(typename PROCEDURE::ExpectedData* spot)
  {
    if(TWarnings) printf("Error: Queue does not implement completeSpot\n");
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ void completeSpot(typename PROCEDURE::ExpectedData* spot)
  {
    if(TWarnings) printf("Error: Queue does not implement completeSpot\n");
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = -1)
  {
    if(TWarnings) printf("Error: Queue does not implement dequeue\n");
    return 0;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
  {
    if(TWarnings) printf("Error: Queue does not implement dequeueSelected\n");
    return 0;
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = -1)
  {
    if(TWarnings) printf("Error: Queue does not implement dequeuePointer\n");
    return 0;
  }

  template<class PROCEDURE>
  __inline__ __device__ int reserveRead(int maxNum = -1)
  {
    if(TWarnings) printf("Error: Queue does not implement reserveRead\n");
    return 0;
  }
  template<class PROCEDURE>
  __inline__ __device__ int startRead(void*& data, int num)
  {
    if(TWarnings) printf("Error: Queue does not implement startRead\n");
    return -1;
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead(int id,  int num)
  {
    if(TWarnings) printf("Error: Queue does not implement finishRead\n");
  }

  __inline__ __device__ void numEntries(int* counts)
  {
    if(TWarnings) printf("Error: Queue does not implement numEntries\n");
  }


  __inline__ __device__ void record()
  {
    if(TWarnings) printf("Error: Queue does not implement record\n");
  }
  __inline__ __device__ void reset()
  {
   if(TWarnings)  printf("Error: Queue does not implement reset\n");
  }


  __inline__ __device__ void workerStart()
  { }
  __inline__ __device__ void workerMaintain()
  { }
  __inline__ __device__ void workerEnd()
  { }
  __inline__ __device__ void globalMaintain()
  { }

  static std::string name()
  {
    if(TWarnings) 
      return "UnnamedQueuing";
    else
      return "";
  }
};

template<class ProcedureInfo, template<class /*PI*/> class RealQueue, template<class> class MatchMaker>
class QueueEnqueueWrapper : public RealQueue<ProcedureInfo>
{
public:
   template<class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    return RealQueue<ProcedureInfo>:: template enqueueInitial < typename MatchMaker<PROCEDURE>::Match>(data);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData *data) 
  {
    return RealQueue<ProcedureInfo>:: template enqueueInitial < typename MatchMaker<PROCEDURE>::Match>(data);
  }

  template<class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {
    return RealQueue<ProcedureInfo>:: template enqueue < typename MatchMaker<PROCEDURE>::Match>(data);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    return RealQueue<ProcedureInfo>:: template enqueue <threads, typename MatchMaker<PROCEDURE>::Match>(data);
  }
};

template<class TAdditionalData, bool TWarnings = true>
class BasicQueue
{
public:

  __inline__ __device__ void init() 
  {
    printf("Warning: BasicQueue does not implement init\n");
  }
  
  template<class Data>
  __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additionalData) 
  {
    printf("Error: BasicQueue does not implement enqueueInitial\n");
    return false;
  }

  template<int Threads, class Data>
  __inline__ __device__ bool enqueueInitial(Data* data, TAdditionalData additionalData) 
  {
    printf("Error: BasicQueue does not implement enqueueInitial\n");
    return false;
  }

  template<class Data>
  __inline__ __device__ bool enqueue(Data data, TAdditionalData additionalData) 
  {
    printf("Error: BasicQueue does not implement enqueue\n");
    return false;
  }

  template<class Data>
  __inline__ __device__ Data* reserveSpot(TAdditionalData additionalData)
  {
    if(TWarnings) printf("Error: BasicQueue does not implement reserveSpot\n");
    return 0;
  }

  template<int threads, class Data>
  __inline__ __device__ Data* reserveSpot(TAdditionalData additionalData)
  {
    if(TWarnings) printf("Error: BasicQueue does not implement reserveSpot<Threads>\n");
    return 0;
  }

  template<class Data>
  __inline__ __device__ void completeSpot(Data* spot)
  {
    if(TWarnings) printf("Error: BasicQueue does not implement completeSpot\n");
  }

  template<int threads, class Data>
  __inline__ __device__ void completeSpot(Data* spot)
  {
    if(TWarnings) printf("Error: BasicQueue does not implement completeSpot\n");
  }

  template<int threads, class Data>
  __inline__ __device__ bool enqueue(Data* data, TAdditionalData additionalData) 
  {
    printf("Error: BasicQueue does not implement enqueue<Threads>\n");
    return false;
  }

  __inline__ __device__ int dequeue(void* data, TAdditionalData* addtionalData, int maxnum)
  {
    printf("Error: BasicQueue does not implement dequeue\n");
    return 0;
  }

  __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    printf("Error: BasicQueue does not implement reserveRead\n");
    return 0;
  }
  __inline__ __device__ int startRead(void*& data, TAdditionalData* addtionalData, int pos, int num)
  {
    printf("Error: BasicQueue does not implement startRead\n");
    return -1;
  }
  __inline__ __device__ void finishRead(int id, int num)
  {
    printf("Error: BasicQueue does not implement finishRead\n");
  }

  template<class SortInfo>
  __inline__ __device__  bool sort( unsigned int threads)
  {
    printf("Error: BasicQueue does not implement sort\n");
    return false;
  }

  static std::string name()
  {
    return "UnnamedBasicQueue";
  }
};

template<>
class BasicQueue<void>
{
public:
   __inline__ __device__ void init() 
  {
    printf("Warning: BasicQueue does not implement init\n");
  }
  
  template<class Data>
  __inline__ __device__ bool enqueueInitial(Data data) 
  {
    printf("Error: BasicQueue does not implement enqueueInitial\n");
    return false;
  }

  template<int threads, class Data>
  __inline__ __device__ bool enqueueInitial(Data* data) 
  {
    printf("Error: BasicQueue does not implement enqueueInitial*\n");
    return false;
  }

  template<class Data>
  __inline__ __device__ bool enqueue(Data data) 
  {
    printf("Error: BasicQueue does not implement enqueue\n");
    return false;
  }

  template<int threads, class Data>
  __inline__ __device__ bool enqueue(Data* data) 
  {
    printf("Error: BasicQueue does not implement enqueue<Threads>\n");
    return false;
  }

  template<class Data>
  __inline__ __device__ Data* reserveSpot()
  {
    printf("Error: BasicQueue does not implement reserveSpot\n");
    return 0;
  }

  template<int threads, class Data>
  __inline__ __device__ Data* reserveSpot()
  {
    printf("Error: BasicQueue does not implement reserveSpot<Threads>\n");
    return 0;
  }

  template<class Data>
  __inline__ __device__ void completeSpot(Data* spot)
  {
    printf("Error: BasicQueue does not implement completeSpot\n");
  }

  template<int threads, class Data>
  __inline__ __device__ void completeSpot(Data* spot)
  {
    printf("Error: BasicQueue does not implement completeSpot\n");
  }

  __inline__ __device__ int dequeue(void* data, int maxnum)
  {
    printf("Error: BasicQueue does not implement dequeue\n");
    return 0;
  }

  __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    printf("Error: BasicQueue does not implement reserveRead\n");
    return 0;
  }
  __inline__ __device__ int startRead(void*& data, int pos, int num)
  {
    printf("Error: BasicQueue does not implement startRead\n");
    return -1;
  }
  __inline__ __device__ void finishRead(int id, int num)
  {
    printf("Error: Queue does not implement finishRead\n");
  }

  template<class SortInfo>
  __inline__ __device__ bool sort( unsigned int threads)
  {
    printf("Error: BasicQueue does not implement sort\n");
    return false;
  }

  static std::string name()
  {
    return "UnnamedBasicQueue";
  }
};


typedef Queue<false> ZeroQueue;

template<class ProcInfo>
class  IgnoreQueue : public ZeroQueue { };


template<class Q>
__global__ void initQueue(Q* q)
{
  q->init();
}

template<unsigned int Size>
class Min16
{
public:
  static const unsigned int Compute = Size;
};


#define _Min16Macro(Size) \
template<> \
class Min16<Size> \
{ \
public:  \
  static const unsigned int Compute = 16U;  \
};

_Min16Macro(0U)
_Min16Macro(1U)
_Min16Macro(2U)
_Min16Macro(3U)
_Min16Macro(4U)
_Min16Macro(5U)
_Min16Macro(6U)
_Min16Macro(7U)
_Min16Macro(8U)
_Min16Macro(9U)
_Min16Macro(10U)
_Min16Macro(11U)
_Min16Macro(12U)
_Min16Macro(13U)
_Min16Macro(14U)
_Min16Macro(15U)

#undef _Min16Macro
