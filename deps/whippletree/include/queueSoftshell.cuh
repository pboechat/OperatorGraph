#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"


  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
  class QueueSoftshell : public ::BasicQueue, protected ::DistributedStorage<TElementSize, TAdditionalData, TQueueSize>
  {
    static const uint ElementSize = TElementSize;
    static const uint QueueSize = TQueueSize;

    struct Protection
    {
      volatile uint state;
      __device__ void init() 
      {
        state = 0;
      }
      __device__ void acquire() 
      {
        while(state != 0)  __threadfence();
      }
      __device__ void ready() 
      {
        state = 1;
      }
      __device__ void read() 
      {
        while(state != 1)  __threadfence();
      }
      __device__ void free() 
      {
        state = 0;
      }
    };


    Protection protections[QueueSize];

    int count;
    uint front, back;
  public:
    static std::string name()
    {
      std::string n("SoftshellEqual");
      //std::stringstream sstr;
      //sstr << "SoftshellEqual";
      if(TWarpOptimization)
        n += "Warpoptimized";
      //  sstr << "Warpoptimized";
      n = n + "[" + std::to_string(ElementSize) + "/" + std::to_string(QueueSize) + "]";
      //sstr << "[" << ElementSize << "/" << QueueSize << "]";
      //return sstr.str();
      return n;
    }

    __inline__ __device__ void init() 
    {
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
        count = 0, front = 0, back = 0;
      for(uint i = lid; i < QueueSize; i+=blockDim.x*gridDim.x)
        protections[i].init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data const * data, TAdditionalData const * additionalData = 0) 
    {
      return enqueue<Data>(data, additionalData);
    }

     template<class Data>
    __device__ bool enqueue(Data const * data, TAdditionalData const * additionalData = 0) 
    {        
#if __CUDA_ARCH__ >= 300
      if(TWarpOptimization)
      {
        //combine
        uint mask = __ballot(1);
        uint ourcount = __popc(mask);
        uint mypos = __popc(Softshell::lanemask_lt() & mask);

        uint wpos = 0;

        if(mypos == 0)
        {
          int old = atomicAdd(&count, ourcount);
          if(old + ourcount >= QueueSize)
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements!\n");
              Softshell::trap();
            }
            atomicSub(&count, ourcount);
            wpos = -1;
          }
          else
            wpos = atomicAdd(&back, ourcount);
        }

        //get source
        int src = __ffs(mask)-1;
        wpos = __shfl((int)wpos, src);
        if(wpos == -1)
          return false;
     
        uint pos = (wpos + mypos)%QueueSize;

        protections[pos].acquire();
        writeData(data, additionalData, pos);
        __threadfence();
        protections[pos].ready();
      }
      else
#endif
      {
        int old = atomicAdd(&count, 1);
        if(old + 1 >= QueueSize)
        {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements!\n");
              Softshell::trap();
            }
            return false;
        }
        uint pos = atomicAdd(&back, 1) % QueueSize;

        protections[pos].acquire();
        writeData(data, additionalData, pos);
        __threadfence();
        protections[pos].ready();
      }
      return true;
    }

    __inline__ __device__ int dequeue(void* data, int num, TAdditionalData * additionalData = 0)
    {
      __shared__ uint offset, take;

      if(threadIdx.x == 0)
      {      
        int c = atomicSub(&count, num);
        if(c < num)
        {
          atomicAdd(&count, min(num,num - c);
          num = max(c, 0);
        }

        take = num;

        if(num > 0)
          offset = atomicAdd(&front, num);
      }
      __syncthreads();

      if(threadIdx.x < take)
      {
        uint pos = offset + threadIdx.x;
        protections[pos].read();
        readData(reinterpret_cast<uint*>(&data) + ElementSize*threadIdx.x, additionalData + threadIdx.x, offset + threadIdx.x);
        __threadfence();
        protections[pos].free();
      }
      return take;
    }

    __inline__ __device__ int size() const
    {
      return count;
    }
  };
