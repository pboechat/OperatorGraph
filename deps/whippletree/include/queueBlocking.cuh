#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"



  template<uint TQueueSize, bool TWarpOptimization = true, bool TAssertOnOverflow = true, int Backoff = 0>
  class QueueBlockingStub
  {
  protected:
    static const uint QueueSize = TQueueSize;

    volatile uint front;
    volatile uint back;
    uint mutex;

    template<int TthreadsPerElement>
    __inline__ __device__  int2 enqueuePrep(int2 last)
    {
      if(TWarpOptimization)
      {
        //combine
        uint mask = __ballot(1);
        uint ourcount = __popc(mask)/TthreadsPerElement;
        int mypos = __popc(Softshell::lanemask_lt() & mask);

        int wpos = -2;

        if(mypos == 0)
        {
          if(atomicExch(&mutex, 1) != 0)
          { 
            __threadfence();
            if(Backoff)
              backoff<Backoff>(last.y+1);
          }
          else
          {
            wpos = back;
            if(wpos + ourcount - front > QueueSize)
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements!\n");
                Softshell::trap();
              }
              atomicExch(&mutex, 0);
              wpos = -1;
            }
            else
              back = wpos + ourcount;
          }
        }

        int src = __ffs(mask)-1;
        //wpos = __shfl(wpos, src);
        wpos = warpBroadcast<32>(wpos, src);

        if(wpos < 0)
          return make_int2(wpos, last.y + 1);
        return  make_int2(wpos + mypos/TthreadsPerElement,ourcount);
      }
      else
      {
        int mypos = Softshell::laneid() % TthreadsPerElement;
        int2 ret_value = make_int2(-2,last.y+1);
        if(TthreadsPerElement == 1 || mypos == 0)
        {
          if(atomicExch(&mutex, 1) != 0)
          { 
            __threadfence();
            if(Backoff)
              backoff<Backoff>(last.y+1);
          }
          else
          {
            uint pos = back;
            if(pos + 1 - front > QueueSize)
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements!\n");
                Softshell::trap();
              }
              atomicExch(&mutex, 0);
              ret_value.x = -1;
            }
            else
            {
              back = pos + 1;
              ret_value = make_int2(pos,1);
            }
          }
        }
        if(TthreadsPerElement > 1)
        {
          ret_value.x = warpBroadcast<TthreadsPerElement>(ret_value.x, 0);
          ret_value.y = warpBroadcast<TthreadsPerElement>(ret_value.y, 0);
            //ret_value.x = __shfl(ret_value.x, 0, TthreadsPerElement);
            //ret_value.y = __shfl(ret_value.y, 0, TthreadsPerElement);
        }
        return ret_value;
      }
    
    }

    template<int TthreadsPerElement>
    __inline__ __device__  void enqueueEnd(int2 pos_num)
    {
      if(pos_num.x == -1)
        return;
      if(TWarpOptimization)
      {
        int mypos = __popc(Softshell::lanemask_lt() & __ballot(1));
        if(mypos == 0)
          atomicExch(&mutex, 0);
      }
      else
      {
        if(TthreadsPerElement == 1)
          atomicExch(&mutex, 0);
        else
        {
          int mypos = __popc(Softshell::lanemask_lt() & __ballot(1));
          if(mypos == 0)
            atomicExch(&mutex, 0);
        }
      }
    }

    __inline__ __device__ uint2 dequeuePrep(int num)
    {
      __shared__ uint2 offset_take;

      if(threadIdx.x == 0)
      {            
        while(atomicExch(&mutex, 1))
        {
          __threadfence();
          int b = 0;
          if(Backoff)
            backoff<Backoff>(++b);
        }
        uint pos = front;
        uint canTake = min(num, back - pos);
        if(canTake == 0)
        {
          atomicExch(&mutex, 0);
          offset_take.y = 0;
        }
        else
        {
          offset_take.y = canTake;
          offset_take.x = pos;
          front = pos + canTake;
        }
      }

      __syncthreads();
      return offset_take;
    }

    __inline__ __device__ void dequeueEnd(uint2 offset_take)
    {
      if(threadIdx.x == 0 && offset_take.y != 0)
        atomicExch(&mutex, 0);
    }

    __inline__ __device__ void init() 
    {
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
        mutex = 0, front = 0, back = 0;
    }

    static std::string name()
    {
      return std::string("EqualBlocking") + (Backoff?std::string("Backoff") + std::to_string((long long)Backoff):"")  + (TWarpOptimization?"Warpoptimized":"");
    }

  public:

    __inline__ __device__ int size() const
    {
      return back - front;
    }
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true,  int TBackoff = 0>
  class QueueEqualBlocking : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> >
  {
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow,  int TBackoff>
  class QueueEqualBlocking<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow, TBackoff> : public QueueBuilder<TElementSize, TQueueSize, void, QueueBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, QueueStorage<TElementSize, void, TQueueSize> >
  {
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true,  int TBackoff = 0>
  class QueueUnequalBlocking : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, AllocStorage<TElementSize, TAdditionalData, TQueueSize, false> >
  {
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow,  int TBackoff>
  class QueueUnequalBlocking<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow, TBackoff> : public QueueBuilder<TElementSize, TQueueSize, void, QueueBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, AllocStorage<TElementSize, void, TQueueSize, false> >
  {
  };



template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlocking_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingOpt_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, true,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlocking_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingOpt_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, true,true> { };


template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingNoOverflow_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingNoOverflowOpt_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, true,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingNoOverflow_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingNoOverflowOpt_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, true,false> { };


template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingBackoff_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, false,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingBackoffOpt_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, true,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingBackoff_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, false,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingBackoffOpt_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, true,true, 1> { };


template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingBackoffNoOverflow_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, false,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualBlockingBackoffNoOverflowOpt_t : public QueueEqualBlocking<TElementSize, TQueueSize, TAdditionalData, true,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingBackoffNoOverflow_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, false,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalBlockingBackoffNoOverflowOpt_t : public QueueUnequalBlocking<TElementSize, TQueueSize, TAdditionalData, true,false, 1> { };


 template<uint TQueueSize, bool TWarpOptimization = true, bool TAssertOnOverflow = true, int Backoff = 0>
  class QueueDualBlockingStub
  {
  protected:
    static const uint QueueSize = TQueueSize;

    volatile uint front;
    volatile uint back;
    uint mutexBack;
    uint mutexFront;

    template<int TthreadsPerElement>
    __inline__ __device__  int2 enqueuePrep(int2 last)
    {
      if(TWarpOptimization)
      {
        //combine
        uint mask = __ballot(1);
        uint ourcount = __popc(mask)/TthreadsPerElement;
        int mypos = __popc(Softshell::lanemask_lt() & mask);

        int wpos = -2;

        if(mypos == 0)
        {
          if(atomicExch(&mutexBack, 1) != 0)
          { 
            __threadfence();
             if(Backoff)
              backoff<Backoff>(last.y+1);
          }
          else
          {
            wpos = back;
            if(wpos + ourcount - front > QueueSize)
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements!\n");
                Softshell::trap();
              }
              atomicExch(&mutexBack, 0);
              wpos = -1;
            }
          }
        }

        int src = __ffs(mask)-1;
        //wpos = __shfl(wpos, src);
        wpos = warpBroadcast<32>(wpos, src);

        if(wpos < 0)
          return make_int2(wpos,last.y+1);
        return  make_int2(wpos + mypos/TthreadsPerElement,ourcount);
      }
      else
      {
        int mypos = Softshell::laneid() % TthreadsPerElement;
        int2 ret_value = make_int2(-2,last.y+1);
        if(TthreadsPerElement == 1 ||  mypos == 0)
        {
          if(atomicExch(&mutexBack, 1) != 0)
          { 
            __threadfence();
            if(Backoff)
              backoff<Backoff>(last.y+1);
          }
          else
          {
            uint pos = back;
            if(pos + 1 - front > QueueSize)
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements!\n");
                Softshell::trap();
              }
              atomicExch(&mutexBack, 0);
              ret_value.x = -1;
            }
            else
            {
              ret_value = make_int2(pos,1);
            }
          }
        }
        if(TthreadsPerElement > 1)
        {
          ret_value.x = warpBroadcast<TthreadsPerElement>(ret_value.x, 0);
          ret_value.y = warpBroadcast<TthreadsPerElement>(ret_value.y, 0);
            //ret_value.x = __shfl(ret_value.x, 0, TthreadsPerElement);
            //ret_value.y = __shfl(ret_value.y, 0, TthreadsPerElement);
        }
        return ret_value;
      }
    }

    template<int TthreadsPerElement>
    __inline__ __device__  void enqueueEnd(int2 pos_num)
    {
      if(pos_num.x == -1)
        return;
      if(TWarpOptimization)
      {
        int mypos = __popc(Softshell::lanemask_lt() & __ballot(1));
        if(mypos == 0)
        {
          back = pos_num.x + pos_num.y;
          __threadfence();
          atomicExch(&mutexBack, 0);
        }
      }
      else
      {
        int mypos = __popc(Softshell::lanemask_lt() & __ballot(1));
        if(TthreadsPerElement == 1 || mypos == 0)
        {
          back = pos_num.x + 1;
          __threadfence();
          atomicExch(&mutexBack, 0);
        }
      }
    }

    __inline__ __device__ uint2 dequeuePrep(int num)
    {
      __shared__ uint2 offset_take;

      if(threadIdx.x == 0)
      {            
        while(atomicExch(&mutexFront, 1))
        {
          __threadfence();
          int b = 0;
          if(Backoff)
            backoff<Backoff>(++b);
        }
        uint pos = front;
        uint canTake = min(num, back - pos);
        if(canTake == 0)
        {
          atomicExch(&mutexFront, 0);
          offset_take.y = 0;
        }
        else
        {
          offset_take.y = canTake;
          offset_take.x = pos;
        }
      }

      __syncthreads();
      return offset_take;
    }

    __inline__ __device__ void dequeueEnd(uint2 offset_take)
    {
      if(threadIdx.x == 0 && offset_take.y != 0)
      {
        front = offset_take.x + offset_take.y;
         __threadfence();
        atomicExch(&mutexFront, 0);
      }
    }

    __inline__ __device__ void init() 
    {
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
        mutexFront = 0, mutexBack = 0, front = 0, back = 0;
    }

    static std::string name()
    {
      return std::string("EqualDualBlocking") + (Backoff?std::string("Backoff") + std::to_string((long long)Backoff):"") + (TWarpOptimization?"Warpoptimized":"");
    }

  public:

    __inline__ __device__ int size() const
    {
      return back - front;
    }
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true,  int TBackoff = 0>
  class QueueEqualDualBlocking : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDualBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> >
  {
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow,  int TBackoff>
  class QueueEqualDualBlocking<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow, TBackoff> : public QueueBuilder<TElementSize, TQueueSize, void, QueueDualBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, QueueStorage<TElementSize, void, TQueueSize> >
  {
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true,  int TBackoff = 0>
  class QueueUnequalDualBlocking : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDualBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, AllocStorage<TElementSize, TAdditionalData, TQueueSize, false> >
  {
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow, int TBackoff>
  class QueueUnequalDualBlocking<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow, TBackoff> : public QueueBuilder<TElementSize, TQueueSize, void, QueueDualBlockingStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, TBackoff>, AllocStorage<TElementSize, void, TQueueSize, false> >
  {
  };
  
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlocking_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingOpt_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlocking_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingOpt_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,true> { };

template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingNoOverflow_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingNoOverflowOpt_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingNoOverflow_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingNoOverflowOpt_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,false> { };

template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingBackoff_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingBackoffOpt_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingBackoff_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,true, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingBackoffOpt_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,true, 1> { };


template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingBackoffNoOverflow_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueEqualDualBlockingBackoffNoOverflowOpt_t : public QueueEqualDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingBackoffNoOverflow_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, false,false, 1> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueUnequalDualBlockingBackoffNoOverflowOpt_t : public QueueUnequalDualBlocking<TElementSize, TQueueSize, TAdditionalData, true,false, 1> { };