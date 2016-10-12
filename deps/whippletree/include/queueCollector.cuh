#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"

template<uint TQueueSize,  bool TAssertOnOverflow = true, bool TWarpOptimization = true, bool ConcurencyGuarantee = false, bool OverflowUnderflowCheck = true>
class QueueCollectorStub
{
protected:
  static const uint QueueSize = TQueueSize;

  volatile int readyCount;
  int writeCount, front, back, backCounter;


  __inline__ __device__ void init()
  {
    readyCount = writeCount = front = back = backCounter = 0;
  }

  static std::string name()
  {
    return (TWarpOptimization?std::string("CollectorWarpOptimized"):std::string("Collector")) + ((OverflowUnderflowCheck == false) ? std::string("NoCheck") :  std::string(""));
  }


  template<int TThreadsPerElement>
  __inline__ __device__  int2 enqueuePrep(int2 last)
  {
    if(TWarpOptimization)
    {
      //combine
      uint mask = __ballot(1);
      uint ourcount = __popc(mask)/TThreadsPerElement;
      int mypos = __popc(Softshell::lanemask_lt() & mask);

      int wpos = -1;

      if(mypos == 0)
      {
        if(OverflowUnderflowCheck)
        {
          int t = atomicAdd(&writeCount, ourcount);
          if(t + ourcount  >= TQueueSize)
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements %d %d %d %d\n",writeCount,readyCount,front,back);
              Softshell::trap();
            }
            atomicSub(&writeCount, ourcount);
          }
          else
            wpos = atomicAdd(&back, ourcount);
        }
        else
          wpos =  atomicAdd(&back, ourcount);
      }

      //get source
      int src = __ffs(mask)-1;
      wpos = warpBroadcast<32>(wpos, src);
      //wpos = __shfl(wpos, src);

      if(wpos == -1)
        return make_int2(-1,0);
      return make_int2(wpos + mypos/TThreadsPerElement, ourcount);
    }
    else
    {
      int pos = -1;
      if(TThreadsPerElement == 1 || Softshell::laneid() % TThreadsPerElement == 0)
      {
        if(OverflowUnderflowCheck)
        {
          int t = atomicAdd(&writeCount, 1);
          if(t + 1  >= TQueueSize)
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements %d %d %d %d\n",writeCount,readyCount,front,back);
              //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
              Softshell::trap();
            }
            atomicSub(&writeCount, 1);
          }
          else
            pos = atomicAdd(&back, 1);
        }
        else
          pos = atomicAdd(&back, 1);
      }
      if(TThreadsPerElement > 1)
      {
        pos = warpBroadcast<TThreadsPerElement>(pos, 0);
        //pos = __shfl(pos, 0, TThreadsPerElement);
      }
      if(pos != -1)
        return make_int2(pos, 1);
      else
        return make_int2(pos, 0);
    }
  }

  template<int TthreadsPerElement>
  __inline__ __device__  void enqueueEnd(int2 pos_ourcount)
  {
    if(TWarpOptimization)
    {
      int mypos = __popc(Softshell::lanemask_lt() & __ballot(1));
      if(mypos == 0)
      {
        //this would guarantee that 
        if(ConcurencyGuarantee)
          while(atomicCAS(&backCounter, pos_ourcount.x, pos_ourcount.x + pos_ourcount.y) != pos_ourcount.x) 
            __threadfence();
        if(OverflowUnderflowCheck)
          atomicAdd((int*)&readyCount,pos_ourcount.y);
      }
    }
    else
    if(TthreadsPerElement == 1 || Softshell::laneid() % TthreadsPerElement == 0)
    {
      if(ConcurencyGuarantee)
        while(atomicCAS(&backCounter, pos_ourcount.x, pos_ourcount.x + 1) != pos_ourcount.x) 
          __threadfence();
      if(OverflowUnderflowCheck)
        atomicAdd((int*)&readyCount,1);
    }
  }

  __inline__ __device__ uint2 dequeuePrep(int num)
  {
    return make_uint2(0,0);
  }
  __inline__ __device__ void dequeueEnd(uint2 offset_take)
  {
  }

   __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    __shared__ int num;
    if(threadIdx.x == 0)
    {
      if(OverflowUnderflowCheck)
      {
        int lnum = atomicSub((int*)&readyCount, maxnum);
        if(lnum < maxnum && only_read_all)
        {
          atomicAdd((int*)&readyCount, maxnum);
          num = 0;
        }
        else if(lnum < maxnum)
        {
          atomicAdd((int*)&readyCount, min(maxnum, maxnum - lnum));
          num = max(0,min(lnum, maxnum));
        }
        else
          num = maxnum;
      }
      else
      {
        num = maxnum;
      }
    }
    __syncthreads();
    return num;
  }

  __inline__ __device__ void finishRead(int id, int num)
  {
    if(OverflowUnderflowCheck && threadIdx.x == 0)
    {
      int prev = atomicSub(&writeCount, num);
      //printf("finishread %d %d : %d->%d %d %d %d\n",id,num, prev, prev-num,readyCount,front,back);
    }
  }

public:

  __inline__ __device__ int size() const
  {
    return readyCount;
  }

 
};


template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TAssertOnOverflow = true, bool TWarpOptimization = true,  bool OverflowUnderflowCheck = true>
class QueueCollector : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueCollectorStub<TQueueSize, TAssertOnOverflow, TWarpOptimization, false, OverflowUnderflowCheck>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> >
{
public:
  __inline__ __device__ int dequeue(void* data, TAdditionalData* addtionalData, int maxnum)
  {
    printf("Error: QueueCollector does not implement dequeue!\n");
    return 0;
  }

  __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    return QueueCollectorStub<TQueueSize, TAssertOnOverflow, TWarpOptimization, false, OverflowUnderflowCheck>::reserveRead(maxnum, only_read_all);
  }

  __inline__ __device__ void finishRead(int id, int num)
  {
    return QueueCollectorStub<TQueueSize, TAssertOnOverflow, TWarpOptimization, false, OverflowUnderflowCheck>::finishRead(id, num);
  }

  __inline__ __device__ int startRead(void*& data, TAdditionalData* addtionalData, int pos, int num)
  {
    __shared__ int offset;
    if(num > 0)
    {
      if(threadIdx.x == 0)
        offset = atomicAdd(&front, num);    
      __syncthreads();
      if(pos < num)
        data = readDataPointers(addtionalData + pos, offset + pos);      
    }
    return num;
  }
};
  
template<uint TElementSize, uint TQueueSize, bool TAssertOnOverflow, bool TWarpOptimization, bool OverflowUnderflowCheck>
class QueueCollector<TElementSize, TQueueSize, void, TAssertOnOverflow, TWarpOptimization, OverflowUnderflowCheck> : public QueueBuilder<TElementSize, TQueueSize, void, QueueCollectorStub<TQueueSize, TAssertOnOverflow, TWarpOptimization, false, OverflowUnderflowCheck>, QueueStorage<TElementSize, void, TQueueSize> >
{
  public:
    __inline__ __device__ int dequeue(void* data, int maxnum)
  {
    printf("Error: QueueCollector does not implement dequeue!\n");
    return 0;
  }

  __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    return QueueCollectorStub<TQueueSize, TAssertOnOverflow, TWarpOptimization, false, OverflowUnderflowCheck>::reserveRead(maxnum, only_read_all);
  }

  
  __inline__ __device__ void finishRead(int id, int num)
  {
    return QueueCollectorStub<TQueueSize, TAssertOnOverflow, TWarpOptimization, false, OverflowUnderflowCheck>::finishRead(id, num);
  }

  __inline__ __device__ int startRead(void*& data, int pos, int num)
  {
    __shared__ int offset;
    if(num > 0)
    {
      if(threadIdx.x == 0)
        offset = atomicAdd(&front, num);    
      __syncthreads();
      if(pos < num)
        data = readDataPointers(offset + pos);      
    }
    return num;
  }
};

template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCollectorNoOpt_t : public QueueCollector<TElementSize, TQueueSize, TAdditionalData, true, false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCollector_t : public QueueCollector<TElementSize, TQueueSize, TAdditionalData, true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCollectorNoOverflow_t : public QueueCollector<TElementSize, TQueueSize, TAdditionalData, false> { };

template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCollectorNoOptNoCheck_t : public QueueCollector<TElementSize, TQueueSize, TAdditionalData, true, false, false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCollectorNoCheck_t : public QueueCollector<TElementSize, TQueueSize, TAdditionalData, true, true, false> { };


