#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"


  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
  class QueueSoftshellPointed : public ::BasicQueue, protected ::DistributedStorage<TElementSize, TAdditionalData, TQueueSize>
  {
    static const uint ElementSize = TElementSize;
    static const uint AdditionalSize = (AdditionalDataInfo<TAdditionalData>::size+sizeof(uint)-1)/sizeof(uint);
    static const uint QueueSize = TQueueSize;

    volatile int offsets[QueueSize];
    volatile int sizes[QueueSize];

    int count;
    uint front, back;
  public:
    static std::string name()
    {
      std::string n("QueueSoftshellPointed");
      //std::stringstream sstr;
      //sstr << "QueueSoftshellPointed";
      if(TWarpOptimization)
        n += "Warpoptimized";
        //sstr << "Warpoptimized";
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
        offsets[i] = -1;
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
        }

        //get source
        int src = __ffs(mask)-1;
        wpos = __shfl((int)wpos, src);
        if(wpos == -1)
          return false;

        int offset = alloc(sizeof(Data));
        store(offset, data, additionalData);
        __threadfence();
        
        if(mypos == 0)
            wpos = atomicAdd(&back, ourcount);

        wpos = __shfl((int)wpos, src);
     
        uint pos = (wpos + mypos)%QueueSize;

        sizes[pos] = size;
        offsets[pos] = offset;
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
            atomicSub(&count, 1);
            return false;
        }


        int offset = alloc(sizeof(Data));
        store(offset, data, additionalData);
        __threadfence();

        uint pos = atomicAdd(&back, 1) % QueueSize;
        sizes[pos] = size;
        offsets[pos] = offset;
      }
      return true;
    }

    __inline__ __device__ int dequeue(void* data, int num, TAdditionalData * additionalData = 0)
    {
      __shared__ int take, offset, size;
      int alloffset;
      uint cdata = (uint*)(data);
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
          alloffset = atomicAdd(&front, num);
      }
      __syncthreads();

      for(int taking = 0; taking < take; ++taking)
      {
        if(threadIdx.x == 0)
        {
          int pos = alloffset + taking;
          while((offset = offsets[pos]) == -1)
            __threadfence();
          size = sizes[pos];
        }

       for(int j = threadIdx.x; j < size; j+=blockDim.x)
         cdata[j] = getDataPointer(offset)[j];
       for(int j = threadIdx.x; j < AdditionalSize; j+=blockDim.x)
         additionalData[j] = reinterpret_cast<volatile uint*>(getAdditionalDataPointer(offset))[j];
       
        __threadfence();
        __syncthreads();
        cdata += size;
      }

      if(threadIdx.x < taking)
        offsets[pos + alloffset + threadIdx.x] = -1;
      return take;
    }

    __inline__ __device__ int size() const
    {
      return count;
    }
  };
