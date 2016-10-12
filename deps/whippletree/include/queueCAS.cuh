#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"


 template<uint TQueueSize, bool TWarpOptimization = true, bool TAssertOnOverflow = true, int Backoff = 0>
  class QueueCASStub
  {
  protected:
    static const uint QueueSize = TQueueSize;

    volatile uint backCounter;
    volatile uint frontCounter;
    volatile uint front;
    volatile uint back;
    volatile int reserved;    

    static std::string name()
    {
      return std::string("CASOrdering") + (Backoff?std::string("Backoff") + std::to_string((long long)Backoff):"") + (TWarpOptimization?"Warpoptimized":"");
    }

    __inline__ __device__ void init() 
    {
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
        backCounter = 0, frontCounter = 0, front = 0, back = 0, reserved = 0;
    }

    template<int TthreadsPerElement>
    __inline__ __device__  int2 enqueuePrep(int2 last)
    {
      if(TWarpOptimization)
      {
        //combine
        uint mask = __ballot(1);
        uint ourcount = __popc(mask)/TthreadsPerElement;
        int mypos = __popc(Softshell::lanemask_lt() & mask);

        int wpos = -1;

        if(mypos == 0)
        {
          wpos = atomicAdd((uint*)&backCounter, ourcount);
          while(wpos + ourcount - front > QueueSize)
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements\n");
              //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
              Softshell::trap();
            }
            if(atomicCAS((uint*)&backCounter, wpos + ourcount, wpos) == wpos + ourcount)
            {
              wpos = -1;
              break;
            }
            if(Backoff)
              backoff<Backoff>(1);
          }
        }

        //get source

        int src = __ffs(mask)-1;
        //wpos = __shfl(wpos, src);
        wpos = warpBroadcast<32>(wpos, src);

        if(wpos == -1)
          return make_int2(-1,0);
        return make_int2(wpos + mypos/TthreadsPerElement, ourcount);
      }
      else
      {
        if(TthreadsPerElement == 1)
        {
          int pos = atomicAdd((uint*)&backCounter, 1);
          while(pos - front > QueueSize)
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements!\n");
              Softshell::trap();
            }
            if(atomicCAS((uint*)&backCounter, pos + 1, pos) == pos + 1)
              return make_int2(-1,0);    
          }
          return make_int2(pos,1);
        }
        else
        {
          int pos = -1;
          if(Softshell::laneid() % TthreadsPerElement == 0)
          {
            pos = atomicAdd((uint*)&backCounter, 1);
            while(pos - front > QueueSize)
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements!\n");
                Softshell::trap();
              }
              if(atomicCAS((uint*)&backCounter, pos + 1, pos) == pos + 1)
                pos = -1;
              else if(Backoff)
                backoff<Backoff>(1);
            }
          }
          pos = warpBroadcast<TthreadsPerElement>(pos, 0);
          //pos = __shfl(pos, 0, TthreadsPerElement);
          if(pos == -1)
            return make_int2(-1,0);
          return make_int2(pos,1);
        }
      }
    }

    template<int TthreadsPerElement>
    __inline__ __device__  void enqueueEnd(int2 pos_ourcount)
    {
      if(TWarpOptimization)
      {
        int mypos = __popc(Softshell::lanemask_lt() & __ballot(1));
        if(mypos == 0)
          while(atomicCAS((uint*)&back, pos_ourcount.x, pos_ourcount.x + pos_ourcount.y) != pos_ourcount.x)
          {
            __threadfence();
            if(Backoff)
              backoff<Backoff>(1);
          }
      }
      else
      {
        if(TthreadsPerElement == 1)
        {
          while(atomicCAS((uint*)&back, pos_ourcount.x, pos_ourcount.x + 1) != pos_ourcount.x)
          {
            __threadfence();
            if(Backoff)
              backoff<Backoff>(1);
          }
        }
        else
        {
          if(Softshell::laneid() % TthreadsPerElement == 0)
            while(atomicCAS((uint*)&back, pos_ourcount.x, pos_ourcount.x + 1) != pos_ourcount.x)
            {
              __threadfence();
              if(Backoff)
                backoff<Backoff>(1);
            }
        }
      }
    }

    __inline__ __device__ uint2 dequeuePrep(int num)
    {
      __shared__ uint2 offset_take;

      if(threadIdx.x == 0)
      {            
        uint f = frontCounter;
        uint b = back;
        offset_take.y = 0;
        while(b > f) 
        {
          uint canTake = min(num, b - f);
          uint pos = atomicCAS((uint*)&frontCounter, f, f + canTake);
          if(pos == f)
          {
            offset_take.y = canTake;
            offset_take.x = pos;
            break;
          }
          else
          {
            f = pos;
            b = back;
            __threadfence();
            if(Backoff)
              backoff<Backoff>(1);
          }
        }
      }

      __syncthreads();
      return offset_take;
    }

    __inline__ __device__ void dequeueEnd(uint2 offset_take)
    {
      if(threadIdx.x == 0 && offset_take.y != 0)
      {
        while(atomicCAS((uint*)&front, offset_take.x, offset_take.x + offset_take.y) != offset_take.x ) 
          __threadfence();
      }
    }

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      __shared__ int num;

      if(threadIdx.x == 0)
      {
        uint b = back;
        uint f = frontCounter;
        int r = reserved; 
        int available = ((int)(b - f)) - r;
        if(available <= 0 || (only_read_all && available < maxnum) )
          num = 0;
        else
        {
          int request =  min(available, maxnum);
          r = atomicAdd((int*)&reserved,request) + request;
          //printf("%d %d changed reserved (initreserve): %d->%d\n",blockIdx.x, threadIdx.x, r, r + request);
          b = back;
          f = frontCounter;
          available = ((int)(b - f)) - r;
          if(available < 0)
          {
            int putback = min(-available, request);
            if(only_read_all)
              putback = maxnum;
            r = atomicSub((int*)&reserved, putback);
            //printf("%d %d changed reserved (putback): %d->%d\n",blockIdx.x, threadIdx.x, r, r - putback);
            num = request - putback;
          }
          else
            num = request;
        }
      }
      __syncthreads();
      return num;
    }

    __inline__ __device__ void finishRead(int id, int num)
    {
      if(threadIdx.x == 0 && num != 0)
      {
        while(atomicCAS((uint*)&front, id, id + num) != id )
        {
          __threadfence();
          if(Backoff)
              backoff<Backoff>(1);
        }
      }
    }

  public:

    __inline__ __device__ int size() const
    {
      uint b = back;
      uint f = frontCounter;
      uint r = reserved; 
      if((int)(b - f) <= r) return 0;
      return  b - f - r;
    }

   
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true, int TBackoff = 0>
  class QueueCASOrdering : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueCASStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> >
  {
  public:
    __inline__ __device__ int startRead(void*& data, TAdditionalData* addtionalData, int pos, int num)
    {
      __shared__ int offset;
      if(threadIdx.x == 0)
      {
        offset = atomicAdd((uint*)&frontCounter, num);
        int r = atomicSub((int*)&reserved, num);
        //printf("%d %d changed reserved (startread): %d->%d\n",blockIdx.x, threadIdx.x, r, r - num);
      }
      __syncthreads();

      if(pos < num)
        data = QueueStorage<TElementSize, TAdditionalData, TQueueSize>::readDataPointers(addtionalData + pos, offset + pos);
      return offset;
    }

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return QueueCASStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::reserveRead(maxnum, only_read_all);    
    }

    __inline__ __device__ void finishRead(int id, int num)
    {
     return QueueCASStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::finishRead(id, num);   
    }
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow, int TBackoff>
  class QueueCASOrdering<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow, TBackoff> : public QueueBuilder<TElementSize, TQueueSize, void, QueueCASStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, QueueStorage<TElementSize, void, TQueueSize> >
  {
  public:
    __inline__ __device__ int startRead(void*& data, int pos, int num)
    {
      __shared__ int offset;
      if(threadIdx.x == 0)
      {
        offset = atomicAdd((uint*)&frontCounter, num);
        int r = atomicSub((int*)&reserved, num);
        //printf("%d %d changed reserved (startread): %d->%d\n",blockIdx.x, threadIdx.x, r, r - num);
      }
      __syncthreads();

      if(pos < num)
        data = QueueStorage<TElementSize, void, TQueueSize>::readDataPointers(offset + pos);
      return offset;
    }
    
    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return QueueCASStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::reserveRead(maxnum, only_read_all);    
    }

    __inline__ __device__ void finishRead(int id, int num)
    {
     return QueueCASStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::finishRead(id, num);   
    }
  };
 
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrdering_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingOpt_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, true,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingNoOverflow_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingNoOverflowOpt_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, true,false> { };

 
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingBackoff_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, false,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingBackoffOpt_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, true,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingBackoffNoOverflow_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, false,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueCASOrderingBackoffNoOverflowOpt_t : public QueueCASOrdering<TElementSize, TQueueSize, TAdditionalData, true,false, 1> { };

