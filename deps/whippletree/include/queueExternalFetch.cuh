#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"

template<uint TElementSize, uint TQueueSize, class TAdditionalData >
class QueueExternalFetch : public ::BasicQueue<void>, protected QueueStorage<TElementSize, void, TQueueSize>
{
  static const uint ElementSize = (TElementSize + sizeof(uint) - 1)/sizeof(uint);
  int count;
  int readCount;
  int maxcount;
public:

  __inline__ __device__ void init()
  {
    QueueStorage<TElementSize, void, TQueueSize>::init();
    maxcount = readCount = count = 0;
  }

  static std::string name()
  {
    return "ExternalFetch";
  }

  template<class Data>
  __inline__ __device__ bool enqueueInitial(Data data) 
  {
    int pos = atomicAdd(&maxcount, 1);
    uint info = prepareData(data);
    writeData(data, make_uint2(pos, info));
    return true;
  }

  template<class Data>
  __device__ bool enqueue(Data data) 
  {        
    printf("ERROR QueueExternalFetch does not support enqueue\n");
    return false;
  }

  __inline__ __device__ int dequeue(void* data, int num)
  {
    __shared__ uint2 offset_take;
    if(threadIdx.x == 0)
    {
      int rc = atomicAdd(&readCount, num);
      int available = maxcount - rc;
      if(available < num)
        atomicSub(&readCount, max(0,min(num - available, num)));
      if(available > 0)
        rc = atomicAdd((int*)&count, min(available, num));
      offset_take.x = max(0,rc);
      offset_take.y = max(0,min(available, num));
    }
    __syncthreads();
   
    if(threadIdx.x < offset_take.y)
    {
      this->readData(reinterpret_cast<uint*>(data) + threadIdx.x * ElementSize, offset_take.x + threadIdx.x);
      __threadfence();
    }
    __syncthreads();
    return offset_take.y;
  }

  __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    __shared__ int num;
    if(threadIdx.x == 0)
    {
      int available = maxcount - atomicAdd(&readCount, maxnum);
      if(available < maxnum && only_read_all)
      {
        atomicSub(&readCount, maxnum);
        num = 0;
      }
      else if(available < maxnum)
      {
        atomicSub((int*)&readCount, max(0,min(maxnum, maxnum - available)));
        num = max(0,min(available, maxnum));
      }
      else
        num = maxnum;
    }
    __syncthreads();
    return num;
  }
  __inline__ __device__ int startRead(void*& data, int pos, int num)
  {
    __shared__ int offset;
    if(num > 0)
    {
      if(threadIdx.x == 0)
        offset = atomicAdd((int*)&count, num);    
      __syncthreads();
      if(pos < num)
        data  = this->readDataPointers(offset + pos);      
    }
    return num;
  }

  __inline__ __device__ void finishRead(int id, int num)
  {
  }

  __inline__ __device__ uint size()
  {
    return max(maxcount - readCount, 0);
  }

  __inline__ __device__ void record()
  {
  }
  __inline__ __device__ void reset()
  {
    count = 0;
    readCount = 0;
  }
};
