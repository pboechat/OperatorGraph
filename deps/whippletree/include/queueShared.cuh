#pragma once
#include "queueInterface.cuh"
#include "procedureInterface.cuh"
#include "procinfoTemplate.cuh"
#include "queueHelpers.cuh"
#include "random.cuh"
#include <string>

template<class PROCEDURE, int ProcId, int NumElements, bool TWarpOptimization>
struct SharedBaseQueue
{
  static const int HeaderSize = 4*sizeof(uint);

  uint procId_maxnum;
  volatile int counter;
  uint headerVersatile0;
  uint headerVerstaile1;

  typename PROCEDURE::ExpectedData queueData[NumElements];

  __inline__ __device__ void clean(int tid, int threads) 
  {
    for(int i = tid; i < 4; i+=threads)
      reinterpret_cast<uint*>(this)[i] = 0;
  }
  __inline__ __device__ void writeHeader()
  {
    procId_maxnum = (ProcId << 16) | NumElements;
  }
  __inline__ __device__ int procId() const
  {
    return procId_maxnum >> 16;
  }
  __inline__ __device__ uint numElement() const
  {
    return procId_maxnum & 0xFFFF;
  }
  __inline__ __device__ int num() const
  {
    return min(counter,NumElements);
  }
  __inline__ __device__ int count() const
  {
    return counter;
  }

  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {
    return enqueue<1>(&data);
  }

  template<uint ThreadsPerElement>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    if(TWarpOptimization)
    {
      uint mask = __ballot(1);
      int ourcount = __popc(mask)/ThreadsPerElement;
      if(counter >= NumElements)
        return false;
      int mypos = __popc(Softshell::lanemask_lt() & mask);

      int spos = -1;
      if(mypos == 0)
      {
        spos = atomicAdd((int*)&counter, ourcount);
        int canPut = max(0, min(NumElements - spos, ourcount));
        if(canPut < ourcount)
          atomicSub((int*)&counter, ourcount - canPut);
      }

      int src = __ffs(mask)-1;
      //spos = __shfl(spos, src);
      spos = warpBroadcast<32>(static_cast<unsigned int>(spos), src);

      int qpos = spos + mypos / ThreadsPerElement;
      if(qpos >= NumElements)
        return false;

      //copy TODO: for a multiple of the threadcount we can unroll that..
      for(int i = threadIdx.x % ThreadsPerElement; i < sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint); i += ThreadsPerElement)
        *(reinterpret_cast<uint*>(queueData + qpos) + i) = *(reinterpret_cast<uint*>(data) + i);
      return true;
    }
    else
    {
      if(counter >= NumElements)
        return false;
      int spos = -1;
      if(threadIdx.x % ThreadsPerElement == 0)
      {
        spos = atomicAdd((int*)&counter, 1);
        if(spos >= NumElements)
          atomicSub((int*)&counter, 1);
      }
      if(ThreadsPerElement != 1)
        spos = warpBroadcast<ThreadsPerElement>(spos, 0);
        //spos = __shfl(spos, 0, ThreadsPerElement);

      if(spos >= NumElements)
        return false;

            //copy
      for(int i = threadIdx.x % ThreadsPerElement; i < sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint); i += ThreadsPerElement)
        *(reinterpret_cast<uint*>(queueData) + sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint)*spos + i) = *(reinterpret_cast<uint*>(data) + i);
      return true;
    }
  }

  __inline__ __device__ int dequeue(void* data, int maxnum)
  {
    int n = counter;
    __syncthreads();
    if(threadIdx.x == 0)
      counter = max(0, n - maxnum);
    int take = min(maxnum, n);
    int offset = n - take;

    for(int i = threadIdx.x; i < sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint)*take; i+=blockDim.x)
     *(reinterpret_cast<uint*>(data) + i) = *(reinterpret_cast<uint*>(queueData) + sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint)*offset + i);

    return take;
  }

  __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
  {
    int n = counter;
    if(only_read_all && n < maxnum)
      return 0;
    return max(0,min(n, maxnum));
  }
  __inline__ __device__ int startRead(typename PROCEDURE::ExpectedData*& data, int num)
  {
    int o = counter - num;
    //if(threadIdx.x == 0)
    //    printf("%d startRead %d->%d\n", blockIdx.x, o, num);
    data = queueData + o;
    return o;
  }
  __inline__ __device__ void finishRead(int id, int num)
  {
    __syncthreads();
    int c = counter;
    int additional = (c - (id + num))*sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint);
    //if(threadIdx.x == 0)
    //    printf("%d finishRead %d->%d, move %d\n", blockIdx.x, c, c-num, additional);
    if(additional > 0)
    {
      //we need to copy to the front
      uint* cdata = reinterpret_cast<uint*>(queueData) + id * sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint) + threadIdx.x;
      for(int i = 0; i < additional*sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint); i += blockDim.x)
      {
        uint d = 0;
        if(i + threadIdx.x < additional)
          d = *(cdata + num * sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint) + i);
        __syncthreads();
        if(i + threadIdx.x < additional)
          *(cdata + i) = d;
      }
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
      //int r = atomicSub((int*)&counter, num);
      counter = c - num;
    }
    __syncthreads();
  }

  static std::string name() 
  {
    return std::string("SharedBaseQueue") + (TWarpOptimization?"Warpoptimized":"");
  }
  
};

//template<class ProcedureInfo, int MemSize, bool TWarpOptimization>
//class SharedDynBaseQueue
//{
//  struct Header
//  {
//    volatile uint procId_maxnum;
//    volatile int counter;
//    uint threads_shared;
//    uint maxalloc_elementSize;
//
//    __inline__ __device__ int procId() const
//    {
//      return (procId_maxnum & 0xFFFF) - 1;
//    }
//    __inline__ __device__ uint numElements() const
//    {
//      return (procId_maxnum >> 16)*16/elementSize();
//    }
//    template<bool MultiProcedure>
//    __inline__ __device__ uint fullElements() const
//    {
//      if(MultiProcedure || (threads_shared & 0x80000000u))
//        return blockDim.x / ((threads_shared >> 16) & 0x7FFFu);
//      return 1;
//    }
//    __inline__ __device__ uint threadsPerElement() const
//    {
//      return (threads_shared >> 16) & 0x7FFF ;
//    }
//    __inline__ __device__ uint requiredShared() const
//    {
//      return threads_shared & 0xFFFF;
//    }
//    __inline__ __device__ int elementSize() const
//    {
//      return maxalloc_elementSize & 0xFFFF;
//    }
//    __inline__ __device__ uint MaxAlloc() const
//    {
//      return maxalloc_elementSize >> 16;
//    }
//    
//    /*static __inline__ __device__ uint& procId_maxnum(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return s_data[hpos];
//    }
//    static __inline__ __device__ uint procId(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return (s_data[hpos] & 0xFFFF) - 1;
//    }
//    static __inline__ __device__ uint numElements(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return (s_data[hpos] >> 16) / s_data[hpos+3];
//    }
//    static __inline__ __device__ uint fullElements(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return blockDim.x / (s_data[hpos+2] >> 16);
//    }
//    static __inline__ __device__ uint threadsPerElement(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return (s_data[hpos+2] >> 16);
//    }
//    static __inline__ __device__ uint requiredShared(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return s_data[hpos+2] & 0xFFFF;
//    }
//    static __inline__ __device__ uint elementSize(int hpos)
//    {
//      extern __shared__ uint s_data[];
//      return s_data[hpos+3];
//    }*/
//  };
//
//  template<bool MultiProcedure>
//  __inline__ __device__ int4 prepareTake(int* procId, int maxShared, int minPercent, bool setCount = true)
//  {
//    __shared__ int4 take_offset_elementsize_threadsperelement; 
//    if(threadIdx.x == 0)
//    {
//      take_offset_elementsize_threadsperelement.x = 0;
//      int hpos = 0;
//      while(hpos < MemSize)
//      {
//        Header* h = reinterpret_cast<Header*>(reinterpret_cast<char*>(this) + hpos);
//        if(h->procId_maxnum == 0)
//        {
//          
//          break;
//        }
//        int maxTake = min(maxShared/h->requiredShared(),h->template fullElements<MultiProcedure>());
//        int threshold = max(min(maxTake,h->numElements()*minPercent/100),1);
//        int c = h->counter;
//        if(c >= threshold)
//        {
//          int take = min(c,maxTake);
//          int offset = hpos + sizeof(Header) + (c - take)*h->elementSize();
//          if(setCount)
//            h->counter = c - take;
//          *procId = h->procId();
//          take_offset_elementsize_threadsperelement = make_int4(take, offset, 
//            h->elementSize(), h->threadsPerElement());
//          //if(threadIdx.x == 0)
//          //  printf("%d prepare take found @%d %d elements, taking %d -> offset %d \n", blockIdx.x, hpos, c, take, offset);
//          break;
//        }
//        hpos += (h->procId_maxnum >> 16)*16 + sizeof(Header);
//      }
//    }
//    __syncthreads();
//    return take_offset_elementsize_threadsperelement;
//  }
//
//public:
//
//  static const int Size = MemSize;
//
//  __inline__ __device__ void init() 
//  {
//    for(int i = threadIdx.x; i < MemSize/sizeof(uint); i+=blockDim.x)
//      reinterpret_cast<uint*>(this)[i] = 0;
//  }
//
//  __inline__ __device__ void maintain() 
//  {
//    __syncthreads();
//    //check for empty elements and copy data to the front, clean out back :)
//    int hpos = 0;
//    int clpos = 0;
//    int lasthpos = 0;
//    while(hpos < MemSize)
//    {
//      Header* h = reinterpret_cast<Header*>(reinterpret_cast<char*>(this) + hpos);
//      if(h->procId_maxnum == 0)
//        break;
//      
//      int thissize = (h->procId_maxnum >> 16)*16 + sizeof(Header);
//      //if(threadIdx.x == 0)
//      //    printf("maintain element @ %d is of size %d\n", hpos, thissize);
//      if(h->counter != 0)
//      {
//        //we do not need to wipe this one, but we gotta copy it
//        if(clpos != hpos)
//        {
//          //if(threadIdx.x == 0)
//          //  printf("maintain copy %d,%d->%d,%d\n", hpos, hpos + thissize, clpos, clpos + thissize);
//          for(int i = 0; i < thissize/sizeof(int); i+=blockDim.x)
//          {
//            int c;
//            if(i + threadIdx.x < thissize/sizeof(int))
//              c = *(reinterpret_cast<int*>(this) + hpos/sizeof(int) + i + threadIdx.x);
//            __syncthreads();
//            if(i + threadIdx.x < thissize/sizeof(int))
//              *(reinterpret_cast<int*>(this) + clpos/sizeof(int) + i + threadIdx.x) = c;
//          }
//        }
//        lasthpos = clpos;
//        clpos += thissize;
//      }
//      else
//        lasthpos = hpos;
//      hpos += thissize;
//    }
//    __syncthreads();
//    Header* h = reinterpret_cast<Header*>(reinterpret_cast<char*>(this) + lasthpos);
//    if(threadIdx.x == 0 && h->procId_maxnum != 0)
//    {
//      uint oldnum =  h->numElements();
//      
//      uint maxnum = min(h->MaxAlloc(), (int) ( (MemSize - lasthpos - sizeof(Header)) /  h->elementSize() ) );
//      h->procId_maxnum = (((maxnum*h->elementSize()+15)/16)<<16) | (h->procId_maxnum & 0xFFFF);
//
//      //if(oldnum != maxnum)
//      //  printf("updating max hold for %d %d->%d\n", lasthpos, oldnum, maxnum);
//    }
//    
//
//    //zero the rest
//    //if(threadIdx.x == 0)
//    //  printf("zeroing %d,%d\n", clpos, hpos);
//
//    for(int i = clpos/sizeof(int) + threadIdx.x; i < hpos/sizeof(int); i+=blockDim.x)
//      *(reinterpret_cast<int*>(this) + i) = 0;
//
//    __syncthreads();
//  }
//
//  template<class PROCEDURE>
//  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data, int MaxQueueAlloc) 
//  {
//    return enqueue<PROCEDURE, 1>(&data, MaxQueueAlloc);
//  }
//
//  template<class PROCEDURE, uint ThreadsPerElement>
//  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data, int MaxQueueAlloc) 
//  {
//    //if(threadIdx.x%ThreadsPerElement == 0)
//    //      printf("%d %d enqueue called for %d\n",blockIdx.x, threadIdx.x,PROCEDURE::ProcedureId);
//    extern __shared__ uint s_data[];
//    int mypos = threadIdx.x % ThreadsPerElement;
//    int ourcount = 1;
//
//    if(TWarpOptimization)
//    {
//      uint mask = __ballot(1);
//      ourcount = __popc(mask)/ThreadsPerElement;
//      mypos = __popc(Softshell::lanemask_lt() & mask);
//    }
//
//      
//    int spos = 0;
//    int hpos = 0;
//    int canPut = 0;
//    while(true)
//    {
//      if(mypos == 0)
//      {
//        while(hpos < MemSize)
//        {
//          Header* h = reinterpret_cast<Header*>(s_data + hpos/4);
//          int maxhold = 0;
//          
//          uint cprocsetup =h->procId_maxnum;
//          if( (cprocsetup & 0xFFFF) == PROCEDURE::ProcedureId + 1)
//          {
//            maxhold = (cprocsetup >> 16)*16/sizeof(typename PROCEDURE::ExpectedData);
//            //printf("%d %d enqueue found match (%d) @ %d which can hold %d\n", blockIdx.x, threadIdx.x, PROCEDURE::ProcedureId, hpos, maxhold);
//          }
//          else if(cprocsetup == 0)
//          {
//
//            uint newsize = min(MaxQueueAlloc, (int) ( (MemSize - hpos - sizeof(Header)) / sizeof(typename PROCEDURE::ExpectedData)));
//            if(sizeof(typename PROCEDURE::ExpectedData) % 16 != 0)
//              newsize = newsize*sizeof(typename PROCEDURE::ExpectedData) / 16 * 16 / sizeof(typename PROCEDURE::ExpectedData);
//
//            if(newsize < MaxQueueAlloc/2)
//            {
//              hpos = MemSize;
//              break;
//            }
//            
//            uint procId_maxnum = ((newsize*sizeof(typename PROCEDURE::ExpectedData)/16)<<16) |  (PROCEDURE::ProcedureId + 1);
//            cprocsetup = atomicCAS((uint*)&h->procId_maxnum, 0, procId_maxnum);
//            if(cprocsetup == 0)
//            {
//              //printf("%d %d enqueue setup new entry for %d @ %d which can hold %d\n", blockIdx.x, threadIdx.x, PROCEDURE::ProcedureId, hpos, newsize );
//              maxhold = newsize;
//              h->threads_shared = ((PROCEDURE::ItemInput?1u:0u) << 31) | (getThreadCount<PROCEDURE>() << 16u) | PROCEDURE::sharedMemory;
//              h->maxalloc_elementSize = (MaxQueueAlloc << 16) | sizeof(typename PROCEDURE::ExpectedData);
//              cprocsetup = procId_maxnum;
//            }
//            else if(cprocsetup == procId_maxnum)
//            {
//              //printf("%d %d second try match for %d @ %d which can hold %d \n", blockIdx.x, threadIdx.x, PROCEDURE::ProcedureId, hpos, newsize);
//              maxhold = newsize;
//            }
//          }
//          if(h->counter < maxhold)
//          {
//            //insert
//            spos = atomicAdd((int*)&h->counter, ourcount);
//            canPut = max(0, min(maxhold - spos, ourcount));
//            if(canPut < ourcount)
//            {
//              int was = atomicSub((int*)&h->counter, ourcount - canPut);
//              //printf("%d %d reducing count for %d @ %d by %d: %d->%d \n", blockIdx.x, threadIdx.x,  PROCEDURE::ProcedureId, hpos, ourcount - canPut, was, was -ourcount + canPut);
//            }
//            if(canPut > 0)
//            {
//              //printf("%d %d inserting %d for %d @ %d which was at %d/%d \n", blockIdx.x, threadIdx.x, canPut, PROCEDURE::ProcedureId, hpos, spos, maxhold);
//              break;
//            }
//          }
//          hpos += (cprocsetup >> 16)*16 + sizeof(Header);
//        }
//      }
//      
//      if(TWarpOptimization)
//      {
//
//        uint mask = __ballot(1);
//        int src = __ffs(mask)-1;
//        spos = warpBroadcast<32>(spos, src);
//        //spos = __shfl(spos, src);
//        hpos = warpBroadcast<32>(hpos, src);
//        //hpos = __shfl(hpos, src);
//        canPut = warpBroadcast<32>(canPut, src);
//        //canPut = __shfl(canPut, src);
//    
//      }
//      else  if(ThreadsPerElement != 1)
//      {
//        spos = warpBroadcast<ThreadsPerElement>(spos, 0);
//        //spos = __shfl(spos, 0, ThreadsPerElement);
//        hpos = warpBroadcast<ThreadsPerElement>(hpos, 0);
//        //hpos = __shfl(hpos, 0, ThreadsPerElement);
//        canPut = warpBroadcast<ThreadsPerElement>(canPut, 0);
//        //canPut = __shfl(canPut, 0, ThreadsPerElement);
//      }
//
//      
//      if(mypos / ThreadsPerElement < canPut)
//      {
//        spos = spos + mypos / ThreadsPerElement;
//        //if(threadIdx.x%ThreadsPerElement == 0)
//        //  printf("%d %d enqueue %d %d (canput: %d mypos: %d)\n",blockIdx.x, threadIdx.x, PROCEDURE::ProcedureId, (int)( hpos + sizeof(Header) + spos * sizeof(typename PROCEDURE::ExpectedData)), canPut, mypos );
//
//        //copy
//        uint* pwrite = reinterpret_cast<uint*>(reinterpret_cast<char*>(this) + hpos + sizeof(Header) + spos * sizeof(typename PROCEDURE::ExpectedData));
//        for(int i = threadIdx.x % ThreadsPerElement; i < sizeof(typename PROCEDURE::ExpectedData)/sizeof(uint); i += ThreadsPerElement)
//          *(pwrite + i) = *(reinterpret_cast<uint*>(data) + i);
//        return true;
//      }
//      else if(hpos >= MemSize)
//      {
//        //printf("%d %d no space in shared queue for %d\n",blockIdx.x, threadIdx.x, PROCEDURE::ProcedureId);
//        return false;
//      }
//      //else
//      //{
//      //  printf("%d %d could not insert: %d < %d?\n",blockIdx.x, threadIdx.x, spos, canPut);
//      //}
//
//      Header* h = reinterpret_cast<Header*>(s_data + hpos/4);
//      hpos += (h->procId_maxnum >> 16)*16 + sizeof(Header);
//
//      if(TWarpOptimization)
//      {
//        uint mask = __ballot(1);
//        ourcount = __popc(mask)/ThreadsPerElement;
//        mypos = __popc(Softshell::lanemask_lt() & mask);
//        canPut = 0;
//      }
//    }
//  }
//  
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeue(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  {
//    int4 take_offset_elementsize_threadsperelement = prepareTake<MultiProcedure>(procId, maxShared, minPercent);
//    int take = take_offset_elementsize_threadsperelement.x;
//    if(take > 0)
//    {
//      int offset = take_offset_elementsize_threadsperelement.y;
//      
//      uint* pread = reinterpret_cast<uint*>(reinterpret_cast<char*>(this) + offset);
//      for(int i = threadIdx.x; i < take*take_offset_elementsize_threadsperelement.z/sizeof(uint); i+=blockDim.x)
//       *(reinterpret_cast<uint*>(data) + i) = *(pread + i);
//
//      data = reinterpret_cast<char*>(data) + take_offset_elementsize_threadsperelement.z*(threadIdx.x/take_offset_elementsize_threadsperelement.w);
//    }
//    return take * take_offset_elementsize_threadsperelement.w;
//  }
//
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1, int minPercent = 80)
//  { 
//    //TODO: implement and test
//    return 0;
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int2 dequeueStartRead(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { 
//    int4 take_offset_elementsize_threadsperelement = prepareTake<MultiProcedure>(procId, maxShared, minPercent, false);
//    int take = take_offset_elementsize_threadsperelement.x * take_offset_elementsize_threadsperelement.w;
//    if(take > 0)
//    {
//      int offset = take_offset_elementsize_threadsperelement.y + take_offset_elementsize_threadsperelement.z*(threadIdx.x/take_offset_elementsize_threadsperelement.w);
//      data = reinterpret_cast<char*>(this) + offset;
//      //if(threadIdx.x == 0)
//      //  printf("%d dequeueStartRead %d (%d) %d with offset %d \n", blockIdx.x, take, take_offset_elementsize_threadsperelement.x, take_offset_elementsize_threadsperelement.y, offset);
//    }
//    return make_int2(take, take_offset_elementsize_threadsperelement.y/4);
//  }
//  __inline__ __device__ void finishRead(int id, int num)
//  {
//    id *= 4;
//    //if(threadIdx.x == 0)
//    //   printf("%d finish read %d %d\n",blockIdx.x, id & 0x3FFFFFFF, num);
//
//    __syncthreads();
//    __shared__ int cpy_from, cpy_bytes;
//
//    //find match
//    if(threadIdx.x == 0)
//    {
//      cpy_bytes = 0;
//      int hpos = 0;
//      while(hpos < MemSize)
//      {
//        Header* h = reinterpret_cast<Header*>(reinterpret_cast<char*>(this) + hpos);
//        if(h->procId_maxnum == 0)
//        {
//          //something went wrong!
//          //printf("%d something went wrong in DynSharedQueue:\n\twe wanted to finish read for %d, but at %d no header is found anymore!\n", blockIdx.x, id, hpos);
//          cpy_bytes = 0;
//          break;
//        }
//        int nhpos = hpos + (h->procId_maxnum >> 16)*16 + sizeof(Header);
//        if(id < nhpos)
//        {
//          int c = h->counter;
//          int offsetid = (id - hpos - sizeof(Header));
//          cpy_bytes = c*h->elementSize() - offsetid - num*h->elementSize();
//          //if(threadIdx.x == 0)
//          //  printf("%d finish read found %d with %d -> %d, copy %d\n",blockIdx.x, hpos, c,  c - num, cpy_bytes/h->elementSize());
//
//          //if(cpy_bytes < 0 && threadIdx.x == 0)
//          //{
//          //  int elementsize = h->elementSize();
//          //  printf("%d something went wrong in DynSharedQueue:\n\cpy_bytes is negative:  %d*%d - %d - %d*%d!\n", blockIdx.x, c,elementsize, offsetid, num, elementsize);
//          //}
//          cpy_from = id + num*h->elementSize();
//
//          h->counter = c - num;
//          break;
//        }
//        hpos = nhpos;        
//      }
//    }
//    __syncthreads();
//
//    if(cpy_bytes)
//    {
//      //we need to copy to the front
//      uint* cto = reinterpret_cast<uint*>(this) + id/sizeof(uint) + threadIdx.x;
//      uint* cfrom = reinterpret_cast<uint*>(this) + cpy_from/sizeof(uint) + threadIdx.x;
//      for(int i = 0; i < cpy_bytes/sizeof(uint); i += blockDim.x)
//      {
//        uint d = 0;
//        if(i + threadIdx.x < cpy_bytes/sizeof(uint))
//          d = *(cfrom + i);
//        __syncthreads();
//        if(i + threadIdx.x < cpy_bytes/sizeof(uint))
//          *(cto + i) = d;
//      }
//      __syncthreads();
//    }
//    //if(threadIdx.x == 0)
//    //   printf("%d finish read done\n",blockIdx.x);
//  }
//
//  static std::string name() 
//  {
//    return std::string("SharedDynBaseQueue") + (TWarpOptimization?"Warpoptimized":"");
//  }
//};
//
//template<template<class PROC, int Elements, bool TWarpOptimization> class Q, bool TWarpOptimization, 
//         class Procedure0,  int QueueElements0, class Procedure1 = EmptyProc<1>, int QueueElements1 = 0, class Procedure2 = EmptyProc<1>, int QueueElements2 = 0, class Procedure3 = EmptyProc<1>, int QueueElements3 = 0, 
//         class Procedure4 = EmptyProc<1>, int QueueElements4 = 0, class Procedure5 = EmptyProc<1>, int QueueElements5 = 0, class Procedure6 = EmptyProc<1>, int QueueElements6 = 0, class Procedure7 = EmptyProc<1>, int QueueElements7 = 0, 
//         class Procedure8 = EmptyProc<1>, int QueueElements8 = 0, class Procedure9 = EmptyProc<1>, int QueueElements9 = 0, class Procedure10 = EmptyProc<1>, int QueueElements10 = 0, class Procedure11 = EmptyProc<1>, int QueueElements11 = 0, 
//         class Procedure12 = EmptyProc<1>, int QueueElements12 = 0, class Procedure13 = EmptyProc<1>, int QueueElements13 = 0, class Procedure14 = EmptyProc<1>, int QueueElements14 = 0, class Procedure15 = EmptyProc<1>, int QueueElements15 = 0>
//struct SharedSizeCounter
//{
//  static const int size = (QueueElements0 > 0? (sizeof(Q<Procedure0, QueueElements0, TWarpOptimization>) + 15)/16*16 : 0) + SharedSizeCounter<Q, TWarpOptimization, Procedure1, QueueElements1,  Procedure2, QueueElements2, Procedure3, QueueElements3, Procedure4, QueueElements4, Procedure5, QueueElements5, Procedure6, QueueElements6, Procedure7, QueueElements7, Procedure8, QueueElements8, Procedure9, QueueElements9, Procedure10, QueueElements10, Procedure11, QueueElements11, Procedure12, QueueElements12, Procedure13, QueueElements13, Procedure14, QueueElements14, Procedure15, QueueElements15>::size;
//};
//template<template<class PROC, int Elements, bool TWarpOptimization> class Q, bool TWarpOptimization, 
//         class Procedure0,  class Procedure1, class Procedure2, class Procedure3,
//         class Procedure4, class Procedure5, class Procedure6, class Procedure7,
//         class Procedure8, class Procedure9, class Procedure10, class Procedure11,
//         class Procedure12, class Procedure13, class Procedure14, class Procedure15>
//struct SharedSizeCounter<Q, TWarpOptimization, Procedure0, 0, Procedure1, 0, Procedure2, 0, Procedure3, 0, Procedure4, 0, Procedure5, 0, Procedure6, 0, Procedure7, 0, Procedure8, 0, Procedure9, 0, Procedure10, 0, Procedure11, 0, Procedure12, 0, Procedure13, 0, Procedure14, 0, Procedure15, 0>
//{
//  static const int size = 0;
//};
//

template<int Size>
struct Make16
{
  static const int Res = (Size+15)/16*16;
};

class EndSharedQueue 
{
public:
  typedef void Proc;
  template<class RootOverallNode, int MaxSize, int PrevSize = 0>
  struct Overall
  {
    static const int Size = 0;
    static const int FinalSize = 0;
    static const int FixedSize = 0;
    static const int SumSize = 0;
    static const int CountDynamicSize = 0;
  };
};


template< template<typename> class SQTraits,class Procedure>
struct GetTraitQueueSize
{
  static const int QueueSize = SQTraits<Procedure>::QueueSize;
};

//template<template<typename> class TWrapper, template<typename> class SQTraits, class Procedure>
//struct GetTraitQueueSize<SQTraits,TWrapper<Procedure> > : public GetTraitQueueSize<SQTraits,Procedure>
//{ };



//intermediate element with queue
template<class ProcInfo, int numOverall, int numPeel, template<typename> class SQTraits, int TNumElements>
class SQElementTraitsPeel
{
  public:
  typedef typename Select<ProcInfo,numPeel>::Procedure Proc;
  typedef SQElementTraitsPeel<ProcInfo,numOverall,numPeel+1,SQTraits,GetTraitQueueSize<SQTraits, typename Select<ProcInfo,numPeel+1>::Procedure >::QueueSize> NextSQElement;
  
  template<class RootOverallNode, int MaxSize, int PrevSize = 0>
  struct Overall
  {
    static const int Size = Make16<TNumElements * sizeof(typename Proc::ExpectedData) + SharedBaseQueue<Proc, 0, TNumElements, true>::HeaderSize>::Res;
    static const int NumElements = TNumElements;
    static const int SumSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::SumSize + Size;
    static const int FixedSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::FixedSize + Size;
    static const int CountDynamicSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::CountDynamicSize;
  };

};


// empty element
template<class ProcInfo, int numOverall, int numPeel, template<typename> class SQTraits>
class SQElementTraitsPeel<ProcInfo,numOverall,numPeel,SQTraits,0> : public SQElementTraitsPeel<ProcInfo,numOverall,numPeel+1,SQTraits, GetTraitQueueSize<SQTraits, typename Select<ProcInfo,numPeel+1>::Procedure >::QueueSize>
{ };

// last element with no shared queue
template<class ProcInfo, int numOverall, template<typename> class SQTraits>
class SQElementTraitsPeel<ProcInfo,numOverall,numOverall,SQTraits,0> : public EndSharedQueue
{ 
public:
  typedef void Proc;
};

// last element with shared queue
template<class ProcInfo, int numOverall, template<typename> class SQTraits, int TNumElements>
class SQElementTraitsPeel<ProcInfo,numOverall,numOverall,SQTraits,TNumElements>
{ 
public:
  typedef typename Select<ProcInfo,numOverall>::Procedure Proc;
  typedef EndSharedQueue NextSQElement;
  
  template<class RootOverallNode, int MaxSize, int PrevSize = 0>
  struct Overall
  {
    static const int Size = Make16<TNumElements * sizeof(typename Proc::ExpectedData) + SharedBaseQueue<Proc, 0, TNumElements, true>::HeaderSize>::Res;
    static const int NumElements = TNumElements;
    static const int SumSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::SumSize + Size;
    static const int FixedSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::FixedSize + Size;
    static const int CountDynamicSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::CountDynamicSize;
  };
};

template<class TProc, int TNum, class TNextSizeSelection = EndSharedQueue>
class SQElementFixedNum
{
public:
  typedef TProc Proc;
  typedef TNextSizeSelection NextSQElement;
  
  template<class RootOverallNode, int MaxSize, int PrevSize = 0>
  struct Overall
  {
    static const int Size = Make16<TNum * sizeof(typename TProc::ExpectedData) + SharedBaseQueue<Proc, 0, TNum, true>::HeaderSize>::Res;
    static const int NumElements = TNum;
    static const int SumSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::SumSize + Size;
    static const int FixedSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::FixedSize + Size;
    static const int CountDynamicSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::CountDynamicSize;
  };
};
template<class TProc, int TSize, class TNextSizeSelection = EndSharedQueue>
class SQElementFixedSize
{
public:
  typedef TProc Proc;
  typedef TNextSizeSelection NextSQElement;
  
  template<class RootOverallNode, int MaxSize, int PrevSize = 0>
  struct Overall
  {
    static const int Size =  Make16<TSize>::Res;
    static const int NumElements = (TSize -  SharedBaseQueue<Proc, 0, 4, true>::HeaderSize) / sizeof(typename TProc::ExpectedData);
    static const int SumSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::SumSize + Size;
    static const int FixedSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::FixedSize + Size;
    static const int CountDynamicSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::CountDynamicSize;
  };
};

template<class TProc, int TRemainingSizeRatio, class TNextSizeSelection = EndSharedQueue>
class SQElementDyn
{
public:
  typedef TProc Proc;
  typedef TNextSizeSelection NextSQElement;
  
  template<class RootOverallNode, int MaxSize, int PrevSize = 0>
  struct Overall
  {
    static const int CountDynamicSize = Make16<NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize>::CountDynamicSize + TRemainingSizeRatio>::Res;
    static const int FixedSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize>::FixedSize;
    static const int Size = Make16<((MaxSize - RootOverallNode::FixedSize) / RootOverallNode::CountDynamicSize - SharedBaseQueue<Proc, 0, 4, true>::HeaderSize)/ sizeof(typename TProc::ExpectedData) * sizeof(typename TProc::ExpectedData) + SharedBaseQueue<Proc, 0, 4, true>::HeaderSize>::Res;
    static const int NumElements = (Size -  SharedBaseQueue<Proc, 0, 4, true>::HeaderSize) / sizeof(typename TProc::ExpectedData);
    static const int SumSize = NextSQElement:: template Overall<RootOverallNode, MaxSize, PrevSize + Size>::SumSize + Size;
  };
};


template<class SelectProc, class ThisProc, class BaseQ, class NextSharedQueueElement>
struct SQueueElementSelectAndForward
{
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, BaseQ* useQ, typename SelectProc::ExpectedData data)
  {
    //forward
    return NextSharedQueueElement:: template enqueue<SelectProc>(sQueueStartPointer, data);
  }
  template<int NumThreads>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, BaseQ* useQ, typename SelectProc::ExpectedData* data)
  {
    //forward
    return NextSharedQueueElement:: template enqueue<NumThreads,SelectProc>(sQueueStartPointer, data);
  }

  __inline__ __device__ static void finishRead(char* sQueueStartPointer, BaseQ* useQ, int id, int num)
  {
    //forward
    return NextSharedQueueElement:: template finishRead <SelectProc> (sQueueStartPointer, id, num);
  }
};

template<class MatchProc, class BaseQ, class NextSharedQueueElement>
struct SQueueElementSelectAndForward<MatchProc,MatchProc,BaseQ,NextSharedQueueElement>
{
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, BaseQ* useQ, typename MatchProc::ExpectedData data)
  {
    //enqueue
    return useQ->enqueue(data);
  }
  template<int NumThreads>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, BaseQ* useQ, typename MatchProc::ExpectedData* data)
  {
    //enqueue
    return useQ-> template enqueue<NumThreads>(data);
  }
     __inline__ __device__ static void finishRead(char* sQueueStartPointer, BaseQ* useQ,  int id, int num)
   {
      return useQ -> finishRead(id, num);
   }
};


template<template<typename> class Wrapper, class MatchProc, class BaseQ, class NextSharedQueueElement>
struct SQueueElementSelectAndForward<MatchProc,Wrapper<MatchProc>,BaseQ,NextSharedQueueElement>
{
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, BaseQ* useQ, typename MatchProc::ExpectedData data)
  {
    //enqueue
    return useQ->enqueue(data);
  }
  template<int NumThreads>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, BaseQ* useQ, typename MatchProc::ExpectedData* data)
  {
    //enqueue
    return useQ-> template enqueue<NumThreads>(data);
  }
     __inline__ __device__ static void finishRead(char* sQueueStartPointer, BaseQ* useQ,  int id, int num)
   {
      return useQ -> finishRead(id, num);
   }
};

template<class ProcInfo, class Procedure, int MaxSize, class TSQDescription, class RootOverallNode, bool WarpOptimization, int PrevSize = 0>
class SharedQueueElement 
{
  typedef typename TSQDescription :: Proc MyProc;
  static const int Size = TSQDescription :: template Overall<RootOverallNode, MaxSize, PrevSize> :: Size;
  static const int NumElements = TSQDescription :: template Overall<RootOverallNode, MaxSize, PrevSize> :: NumElements;
  typedef SharedQueueElement<ProcInfo, typename TSQDescription::NextSQElement :: Proc, MaxSize, typename TSQDescription::NextSQElement, RootOverallNode, WarpOptimization, PrevSize + Size> NextSharedQueueElement;
  typedef SharedBaseQueue<MyProc, findProcId<ProcInfo, MyProc>::value, NumElements, WarpOptimization> MyBaseQueue;

  __inline__ __device__ 
  static MyBaseQueue* myQ(char *sQueueStartPointer) 
  {
    return reinterpret_cast<MyBaseQueue* >(sQueueStartPointer + PrevSize);
  }

public: 
  static const int requiredShared = TSQDescription :: template Overall<RootOverallNode, MaxSize, PrevSize> :: SumSize;
  
  static_assert(requiredShared <= MaxSize, "Shared Queue generated from traits is larger than specified max QueueSize");

  __inline__ __device__ static void init(char* sQueueStartPointer)
  {
    myQ(sQueueStartPointer)->clean(threadIdx.x, blockDim.x);
    myQ(sQueueStartPointer)->writeHeader();
    NextSharedQueueElement::init(sQueueStartPointer);
  }
  __inline__ __device__ static void maintain(char* sQueueStartPointer)
  { }


  template<class TProcedure>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, typename TProcedure::ExpectedData data)
  { 
    return SQueueElementSelectAndForward<TProcedure, MyProc, MyBaseQueue, NextSharedQueueElement> ::enqueue(sQueueStartPointer, myQ(sQueueStartPointer), data);
  }

  template<uint ThreadsPerElement, class TProcedure>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, typename TProcedure::ExpectedData* data) 
  { 
    return SQueueElementSelectAndForward<TProcedure,MyProc,MyBaseQueue,NextSharedQueueElement> :: template enqueue<ThreadsPerElement>(sQueueStartPointer, myQ(sQueueStartPointer), data);
  }



  template<bool MultiProcedure>
  __inline__ __device__ static int dequeue(char* sQueueStartPointer, void*& data, int* procId, int maxShared = -1, int minPercent = 80)
  { 
    int maxElements = getElementCount<MyProc,MultiProcedure>();
    if(maxShared != -1)
      maxElements = min(maxElements, maxShared / ((int)sizeof(typename MyProc::ExpectedData) + MyProc::sharedMemory));

    int DequeueThreshold = minPercent*NumElements/100+1;
    int c = myQ(sQueueStartPointer)->count();
    if(c >=  min(maxElements,DequeueThreshold))
    {
      c = myQ(sQueueStartPointer)->dequeue(data, maxElements);
      if(c > 0)
      {
        *procId = MyProc::ProcedureId;
        data = ((uint*)data) + getThreadOffset<MyProc,MultiProcedure>()*sizeof(typename MyProc::ExpectedData);
      }
      return c * getThreadCount<MyProc>();
    }
    return NextSharedQueueElement :: template dequeue<MultiProcedure>(sQueueStartPointer, data, procId, maxShared, minPercent);
  }

  template<bool MultiProcedure>
  __inline__ __device__ static int dequeueSelected(char* sQueueStartPointer, void*& data, int procId, int maxNum = -1, int minPercent = 80)
  {
    int maxElements = getElementCount<MyProc>();
    if(maxNum != -1)
      maxElements = min(maxElements, maxNum);

    int DequeueThreshold = minPercent*NumElements/100+1;
    int c = myQ(sQueueStartPointer)->count();
    if(c >=  min(maxElements,DequeueThreshold))
    {
      c = myQ(sQueueStartPointer)->dequeue(data, maxElements);
      if(c > 0)
      {
        data = ((uint*)data) + getThreadOffset<MyProc>()*sizeof(typename MyProc::ExpectedData);
      }
      return c;
    }
    return NextSharedQueueElement :: template dequeueSelected<MultiProcedure>(sQueueStartPointer, data, procId, maxNum, minPercent);
  }

  template<bool MultiProcedure>
   __inline__ __device__ static int2 dequeueStartRead(char* sQueueStartPointer, void*& data, int* procId, int maxShared = -1, int minPercent = 80)
  { 
    int maxElements = getElementCount<MyProc, MultiProcedure>();
    if(maxShared != -1)
      maxElements = min(maxElements, MyProc::sharedMemory > 0 ? maxShared / (MyProc::sharedMemory) : blockDim.x);
    int c = myQ(sQueueStartPointer)->count();
    int DequeueThreshold = minPercent*NumElements/100+1;
    if(c >=  min(maxElements,DequeueThreshold))
    {
      c = myQ(sQueueStartPointer)->reserveRead(maxElements);
      int id = 0;
      if(c > 0)
      {
        typename MyProc::ExpectedData* p;
        id = myQ(sQueueStartPointer)->startRead(p, c);
        c = c * getThreadCount<MyProc>();
        data = reinterpret_cast<void*>(p + getThreadOffset<MyProc,MultiProcedure>());
        procId[0] = findProcId<ProcInfo,MyProc>::value; 
      }
      return make_int2(c, id);
    }
    return NextSharedQueueElement :: template  dequeueStartRead<MultiProcedure>(sQueueStartPointer, data, procId, maxShared, minPercent);
  }


  template<class TProcedure>
  __inline__ __device__ static void finishRead(char* sQueueStartPointer, int id, int num)
  {   
    SQueueElementSelectAndForward<TProcedure,MyProc,MyBaseQueue,NextSharedQueueElement> :: finishRead(sQueueStartPointer, myQ(sQueueStartPointer), id, num);
  }

  static std::string name()
  { 
    return std::to_string((long long)findProcId<ProcInfo,MyProc>::value) + "(" + std::to_string((long long)NumElements) + ")" + "," + NextSharedQueueElement :: name();
  }
};


// specialization for end of shared queue
template<class ProcInfo, int MaxSize,  class TSQDescription, class RootOverallNode, bool WarpOptimization, int PrevSize>
class SharedQueueElement<ProcInfo, void, MaxSize, TSQDescription,RootOverallNode,WarpOptimization,PrevSize>
{
public: 
  static const int requiredShared = 0;
  __inline__ __device__ static void init(char* sQueueStartPointer) { }
  __inline__ __device__ static void maintain(char* sQueueStartPointer) { }
  template<class Procedure>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, typename Procedure::ExpectedData otherdata) { return false; }
  template<uint ThreadsPerElement, class Procedure>
  __inline__ __device__ static bool enqueue(char* sQueueStartPointer, typename Procedure::ExpectedData* data) { return false; }
  template<bool MultiProcedure>
  __inline__ __device__ static int dequeue(char* sQueueStartPointer, void*& data, int* procId, int maxShared = -1, int minPercent = 80) { return 0; }
  template<bool MultiProcedure>
  __inline__ __device__ static int dequeueSelected(char* sQueueStartPointer, void*& data, int procId, int maxNum = -1, int minPercent = 80) { return 0; }
  template<bool MultiProcedure>
   __inline__ __device__ static int2 dequeueStartRead(char* sQueueStartPointer, void*& data, int* procId_info, int maxShared = -1, int minPercent = 80) { return make_int2(0,0);}
  template<class Procedure>
  __inline__ __device__ static void finishRead(char* sQueueStartPointer, int id, int num) { }
  static std::string name() { return ""; }
};





//
//template<class Procedure, class CallProcedure, class SharedQ, class NextQ>
//class SharedQueueForwarding
//{
//public:
//  static __inline__ __device__ bool enqueue(SharedQ* sq, NextQ* nq, typename CallProcedure::ExpectedData data) 
//  { 
//    return nq -> template enqueue<CallProcedure> (data);
//  }
//  static __inline__ __device__ bool enqueue(SharedQ* sq, typename CallProcedure::ExpectedData data) 
//  { 
//    return false;
//  }
//
//  template<int Threads>
//  static __inline__ __device__ bool enqueue(SharedQ* sq, NextQ* nq, typename CallProcedure::ExpectedData* data) 
//  { 
//    return nq -> template enqueue<Threads, CallProcedure> (data);
//  }
//
//  template<int Threads>
//  static __inline__ __device__ bool enqueue(SharedQ* sq, typename CallProcedure::ExpectedData* data) 
//  { 
//    return false;
//  }
//
//  static __inline__ __device__ void finishRead(SharedQ* sq, NextQ* nq, int id, int num) 
//  { 
//    nq -> template finishRead<CallProcedure> (id, num);
//  }
//};
//template<class Procedure, class SharedQ, class NextQ>
//class SharedQueueForwarding<Procedure, Procedure, SharedQ, NextQ>
//{
//public:
//  static __inline__ __device__ bool enqueue(SharedQ* sq, NextQ* nq, typename Procedure::ExpectedData data) 
//  { 
//    //put in if possible
//    return sq -> enqueue (data);
//  }
//   static __inline__ __device__ bool enqueue(SharedQ* sq, typename Procedure::ExpectedData data) 
//  { 
//    //put in if possible
//    return sq -> enqueue (data);
//  }
//
//  template<int Threads>
//  static __inline__ __device__ bool enqueue(SharedQ* sq, NextQ* nq, typename Procedure::ExpectedData* data) 
//  { 
//    //put in if possible
//    return sq -> template enqueue <Threads> (data);
//  }
//
//  template<int Threads>
//  static __inline__ __device__ bool enqueue(SharedQ* sq, typename Procedure::ExpectedData* data) 
//  { 
//    //put in if possible
//    return sq -> template enqueue <Threads> (data);
//  }
//
//  static __inline__ __device__ void finishRead(SharedQ* sq, NextQ* nq, int id, int num) 
//  { 
//    sq -> finishRead(id, num);
//  }
//};
//
//template<template<class PROC, int Elements, bool TWarpOptimization> class Q, bool TWarpOptimization,
//         class Procedure0,                 int QueueElements0,      class Procedure1 = EmptyProc<1>,  int QueueElements1 = 0,  class Procedure2 = EmptyProc<1>,  int QueueElements2 = 0,  class Procedure3 = EmptyProc<1>,  int QueueElements3 = 0, 
//         class Procedure4 = EmptyProc<1>,  int QueueElements4 = 0,  class Procedure5 = EmptyProc<1>,  int QueueElements5 = 0,  class Procedure6 = EmptyProc<1>,  int QueueElements6 = 0,  class Procedure7 = EmptyProc<1>,  int QueueElements7 = 0, 
//         class Procedure8 = EmptyProc<1>,  int QueueElements8 = 0,  class Procedure9 = EmptyProc<1>,  int QueueElements9 = 0,  class Procedure10 = EmptyProc<1>, int QueueElements10 = 0, class Procedure11 = EmptyProc<1>, int QueueElements11 = 0, 
//         class Procedure12 = EmptyProc<1>, int QueueElements12 = 0, class Procedure13 = EmptyProc<1>, int QueueElements13 = 0, class Procedure14 = EmptyProc<1>, int QueueElements14 = 0, class Procedure15 = EmptyProc<1>, int QueueElements15 = 0>
//class SharedMultiQueueElement
//{
//  typedef Q<Procedure0, QueueElements0, TWarpOptimization> MyQueue;
//  static const int MySize = (sizeof(MyQueue) + 15)/16*16;
//  typedef SharedMultiQueueElement<Q, TWarpOptimization, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15> NextSharedMultiQueueElement;
//
//  __inline__ __device__  NextSharedMultiQueueElement* next() 
//  {
//    return reinterpret_cast<NextSharedMultiQueueElement*>((reinterpret_cast<char*>(this) + MySize));
//  }
//public:
//    __inline__ __device__ void init()
//  {
//    reinterpret_cast<MyQueue*>(this)->clean(threadIdx.x, blockDim.x);
//    reinterpret_cast<MyQueue*>(this)->writeHeader();
//    next()->init();
//  }
//  __inline__ __device__ void maintain()
//  { }
//  
//  template<class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData otherdata) 
//  { 
//    return SharedQueueForwarding<Procedure0, Procedure, MyQueue, NextSharedMultiQueueElement>::enqueue(reinterpret_cast<MyQueue*>(this), next(), otherdata);
//  }
//  template<uint ThreadsPerElement, class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData* data) 
//  { 
//    return SharedQueueForwarding<Procedure0, Procedure, MyQueue, NextSharedMultiQueueElement>:: template enqueue<ThreadsPerElement>(reinterpret_cast<MyQueue*>(this), next(), data);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeue(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { 
//    int maxElements = getElementCount<Procedure0,MultiProcedure>();
//    if(maxShared != -1)
//      maxElements = min(maxElements, maxShared / ((int)sizeof(typename Procedure0::ExpectedData) + Procedure0::sharedMemory));
//
//    int DequeueThreshold = minPercent*QueueElements0/100+1;
//    int c = reinterpret_cast<MyQueue*>(this)->count();
//    if(c >=  min(maxElements,DequeueThreshold))
//    {
//      c = reinterpret_cast<MyQueue*>(this)->dequeue(data, maxElements);
//      if(c > 0)
//      {
//        *procId = Procedure0::ProcedureId;
//        data = ((uint*)data) + getThreadOffset<Procedure0,MultiProcedure>()*sizeof(typename Procedure0::ExpectedData);
//      }
//      return c * getThreadCount<Procedure0>();
//    }
//    return next() -> dequeue<MultiProcedure>(data, procId, maxShared, minPercent);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1, int minPercent = 80)
//  { 
//    int maxElements = getElementCount<Procedure0>();
//    if(maxNum != -1)
//      maxElements = min(maxElements, maxNum);
//
//    int DequeueThreshold = minPercent*QueueElements0/100+1;
//    int c = reinterpret_cast<MyQueue*>(this)->count();
//    if(c >=  min(maxElements,DequeueThreshold))
//    {
//      c = reinterpret_cast<MyQueue*>(this)->dequeue(data, maxElements);
//      if(c > 0)
//      {
//        data = ((uint*)data) + getThreadOffset<Procedure0>()*sizeof(typename Procedure0::ExpectedData);
//      }
//      return c;
//    }
//    return  next()->dequeueSelected<MultiProcedure>(data, procId, maxNum, minPercent);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int2 dequeueStartRead(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { 
//    int maxElements = getElementCount<Procedure0, MultiProcedure>();
//    if(maxShared != -1)
//      maxElements = min(maxElements, Procedure0::sharedMemory > 0 ? maxShared / (Procedure0::sharedMemory) : blockDim.x);
//    int c = reinterpret_cast<MyQueue*>(this)->count();
//    int DequeueThreshold = minPercent*QueueElements0/100+1;
//    if(c >=  min(maxElements,DequeueThreshold))
//    {
//      c = reinterpret_cast<MyQueue*>(this)->reserveRead(maxElements);
//      int id = 0;
//      if(c > 0)
//      {
//        typename Procedure0::ExpectedData* p;
//        id = reinterpret_cast<MyQueue*>(this)->startRead(p, c);
//        c = c * getThreadCount<Procedure0>();
//        data = reinterpret_cast<void*>(p + getThreadOffset<Procedure0,MultiProcedure>());
//        procId[0] = Procedure0::ProcedureId;
//      }
//      return make_int2(c, id);
//    }
//    return  next() -> dequeueStartRead<MultiProcedure>(data, procId, maxShared, minPercent);
//  }
//  
//  template<class Procedure>
//  __inline__ __device__ void finishRead(int id, int num)
//  { 
//    SharedQueueForwarding<Procedure0, Procedure, MyQueue, NextSharedMultiQueueElement>::finishRead(reinterpret_cast<MyQueue*>(this), next(), id, num);
//  }
//
//  static std::string name()
//  {
//    return std::to_string((long long)Procedure0::ProcedureId) + "(" + std::to_string((long long)QueueElements0) + ")" + "," + NextSharedMultiQueueElement :: name();
//  }
//};
//
//
//template<template<class PROC, int Elements, bool TWarpOptimization> class Q, bool TWarpOptimization,
//         class Procedure0, class Procedure1,int QueueElements1,class Procedure2,int QueueElements2,class Procedure3,int QueueElements3,
//         class Procedure4,int QueueElements4,class Procedure5,int QueueElements5,class Procedure6,int QueueElements6,class Procedure7,int QueueElements7,
//         class Procedure8,int QueueElements8,class Procedure9,int QueueElements9,class Procedure10,int QueueElements10,class Procedure11,int QueueElements11,
//         class Procedure12,int QueueElements12,class Procedure13,int QueueElements13,class Procedure14,int QueueElements14,class Procedure15,int QueueElements15>
//class SharedMultiQueueElement<Q, TWarpOptimization, Procedure0,0,Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>
//{
//    typedef SharedMultiQueueElement<Q, TWarpOptimization, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15> NextSharedMultiQueueElement;
//public:
//    __inline__ __device__ void init()
//  {
//    reinterpret_cast<NextSharedMultiQueueElement*>(this)->init();
//  }
//  __inline__ __device__ void maintain()
//  { }
//  
//  template<class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData otherdata) 
//  { 
//    return reinterpret_cast<NextSharedMultiQueueElement*>(this)-> template enqueue<Procedure>(otherdata);
//  }
//  template<uint ThreadsPerElement, class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData* data) 
//  { 
//    return reinterpret_cast<NextSharedMultiQueueElement*>(this)-> template enqueue<ThreadsPerElement, Procedure>(data);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeue(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { 
//    return reinterpret_cast<NextSharedMultiQueueElement*>(this)-> dequeue<MultiProcedure>(data, procId, maxShared, minPercent);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1, int minPercent = 80)
//  { 
//    return reinterpret_cast<NextSharedMultiQueueElement*>(this)->dequeueSelected<MultiProcedure>(data, procId, maxNum, minPercent);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int2 dequeueStartRead(void*& data, int* procId_info, int maxShared = -1, int minPercent = 80)
//  { 
//    return reinterpret_cast<NextSharedMultiQueueElement*>(this)->dequeueStartRead<MultiProcedure>(data, procId_info, maxShared, minPercent);
//  }
//  template<class Procedure>
//  __inline__ __device__ void finishRead(int id, int num)
//  { 
//    return reinterpret_cast<NextSharedMultiQueueElement*>(this)->  template finishRead<Procedure>(id, num);
//  }
//    
//  static std::string name()
//  {
//    return NextSharedMultiQueueElement :: name();
//  }
//};
//
//template<template<class PROC, int Elements, bool TWarpOptimization> class Q, bool TWarpOptimization, 
//         class Procedure0,  class Procedure1, class Procedure2, class Procedure3,
//         class Procedure4, class Procedure5, class Procedure6, class Procedure7,
//         class Procedure8, class Procedure9, class Procedure10, class Procedure11,
//         class Procedure12, class Procedure13, class Procedure14, class Procedure15>
//class SharedMultiQueueElement<Q, TWarpOptimization, Procedure0, 0, Procedure1, 0, Procedure2, 0, Procedure3, 0, Procedure4, 0, Procedure5, 0, Procedure6, 0, Procedure7, 0, Procedure8, 0, Procedure9, 0, Procedure10, 0, Procedure11, 0, Procedure12, 0, Procedure13, 0, Procedure14, 0, Procedure15, 0>
//{
//public:
//  __inline__ __device__ void init()
//  { }
//  __inline__ __device__ void maintain()
//  { }
//  template<class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData otherdata) 
//  { return false; }
//  template<uint ThreadsPerElement, class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData* data) 
//  { return false; }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeue(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { return 0; }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1, int minPercent = 80)
//  { return 0; }
//  template<bool MultiProcedure>
//   __inline__ __device__ int2 dequeueStartRead(void*& data, int* procId_info, int maxShared = -1, int minPercent = 80)
//  { return make_int2(0,0); }
//  template<class Procedure>
//  __inline__ __device__ void finishRead(int id, int num)
//  {  }
//  static std::string name()
//  { return ""; }
//};
//
//
//
//template<class PROCEDURE, uint MemSize, bool TWarpOptimization = true>
//class SharedSingleQueue : public SharedMultiQueueElement< SharedBaseQueue, TWarpOptimization, PROCEDURE, ((MemSize - SharedBaseQueue<PROCEDURE, 1, TWarpOptimization>::HeaderSize) > sizeof(typename PROCEDURE::ExpectedData) ? (MemSize - SharedBaseQueue<PROCEDURE, 1, TWarpOptimization>::HeaderSize) : sizeof(typename PROCEDURE::ExpectedData)) / 16 * 16 / sizeof(typename PROCEDURE::ExpectedData)>
//{ 
//  //need the size
//  static const int prenumelements = (MemSize - SharedBaseQueue<PROCEDURE, 1, TWarpOptimization>::HeaderSize) / sizeof(typename PROCEDURE::ExpectedData);
//  static const int numelements = prenumelements < 1 ? 1 : prenumelements;
//  SharedBaseQueue<PROCEDURE, numelements, TWarpOptimization> dummyQ;
//};
//
//template<class ProcedureInfo, uint ProcedureNumber, uint MemSize, bool TWarpOptimization = true>
//class SharedSingleSelectedQueue;
//
//#define SharedSingleSelectedQueueDefinition(ID) \
//template<class ProcedureInfo, uint MemSize, bool TWarpOptimization> \
//class SharedSingleSelectedQueue<ProcedureInfo, ID,  MemSize, TWarpOptimization> : public SharedSingleQueue<typename ProcedureInfo::Procedure ## ID, MemSize, TWarpOptimization> \
//{ };
//
//
//SharedSingleSelectedQueueDefinition(0)
//SharedSingleSelectedQueueDefinition(1)
//SharedSingleSelectedQueueDefinition(2)
//SharedSingleSelectedQueueDefinition(3)
//SharedSingleSelectedQueueDefinition(4)
//SharedSingleSelectedQueueDefinition(5)
//SharedSingleSelectedQueueDefinition(6)
//SharedSingleSelectedQueueDefinition(7)
//SharedSingleSelectedQueueDefinition(8)
//SharedSingleSelectedQueueDefinition(9)
//SharedSingleSelectedQueueDefinition(10)
//SharedSingleSelectedQueueDefinition(11)
//SharedSingleSelectedQueueDefinition(12)
//SharedSingleSelectedQueueDefinition(13)
//SharedSingleSelectedQueueDefinition(14)
//SharedSingleSelectedQueueDefinition(15)
//
//#undef SharedSingleSelectedQueueDefinition
//
//
//
//template<bool TWarpOptimization,
//         class Procedure0,                 int QueueElements0,      class Procedure1 = EmptyProc<1>,  int QueueElements1 = 0,  class Procedure2 = EmptyProc<1>,  int QueueElements2 = 0,  class Procedure3 = EmptyProc<1>,  int QueueElements3 = 0, 
//         class Procedure4 = EmptyProc<1>,  int QueueElements4 = 0,  class Procedure5 = EmptyProc<1>,  int QueueElements5 = 0,  class Procedure6 = EmptyProc<1>,  int QueueElements6 = 0,  class Procedure7 = EmptyProc<1>,  int QueueElements7 = 0, 
//         class Procedure8 = EmptyProc<1>,  int QueueElements8 = 0,  class Procedure9 = EmptyProc<1>,  int QueueElements9 = 0,  class Procedure10 = EmptyProc<1>, int QueueElements10 = 0, class Procedure11 = EmptyProc<1>, int QueueElements11 = 0, 
//         class Procedure12 = EmptyProc<1>, int QueueElements12 = 0, class Procedure13 = EmptyProc<1>, int QueueElements13 = 0, class Procedure14 = EmptyProc<1>, int QueueElements14 = 0, class Procedure15 = EmptyProc<1>, int QueueElements15 = 0>
//class SharedMultiQueueSelect : public SharedMultiQueueElement< SharedBaseQueue, TWarpOptimization, 
//         Procedure0,QueueElements0, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>
//{ 
//  static const int OverallSize = SharedSizeCounter<SharedBaseQueue, TWarpOptimization, 
//         Procedure0, QueueElements0, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>::size;
//  //dummy size
//  char dummy[OverallSize > 4 ? OverallSize : 4];
//};
//
//      
//template<class ProcedureInfo, int MemSize, bool TWarpOptimization, 
//         class Procedure0,                 int QueueElements0 = 1,      class Procedure1 = EmptyProc<1>,  int QueueElements1 = 0,  class Procedure2 = EmptyProc<1>,  int QueueElements2 = 0,  class Procedure3 = EmptyProc<1>,  int QueueElements3 = 0, 
//         class Procedure4 = EmptyProc<1>,  int QueueElements4 = 0,  class Procedure5 = EmptyProc<1>,  int QueueElements5 = 0,  class Procedure6 = EmptyProc<1>,  int QueueElements6 = 0,  class Procedure7 = EmptyProc<1>,  int QueueElements7 = 0, 
//         class Procedure8 = EmptyProc<1>,  int QueueElements8 = 0,  class Procedure9 = EmptyProc<1>,  int QueueElements9 = 0,  class Procedure10 = EmptyProc<1>, int QueueElements10 = 0, class Procedure11 = EmptyProc<1>, int QueueElements11 = 0, 
//         class Procedure12 = EmptyProc<1>, int QueueElements12 = 0, class Procedure13 = EmptyProc<1>, int QueueElements13 = 0, class Procedure14 = EmptyProc<1>, int QueueElements14 = 0, class Procedure15 = EmptyProc<1>, int QueueElements15 = 0>
//class SharedMultiQueue : public SharedMultiQueueSelect<TWarpOptimization, 
//         Procedure0,
//         maxOperator<QueueElements0,
//         (MemSize - SharedBaseQueue<Procedure0, 1, TWarpOptimization>::HeaderSize - SharedSizeCounter<SharedBaseQueue, TWarpOptimization, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>::size) / sizeof(typename Procedure0::ExpectedData)>::result,
//         Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>
//{
//  
//};
//
//
//template<class Procedure, 
//         class Procedure0 = EmptyProc<1>,  int QueueElements0 = 0,  class Procedure1 = EmptyProc<1>,  int QueueElements1 = 0,  class Procedure2 = EmptyProc<1>,  int QueueElements2 = 0,  class Procedure3 = EmptyProc<1>,  int QueueElements3 = 0, 
//         class Procedure4 = EmptyProc<1>,  int QueueElements4 = 0,  class Procedure5 = EmptyProc<1>,  int QueueElements5 = 0,  class Procedure6 = EmptyProc<1>,  int QueueElements6 = 0,  class Procedure7 = EmptyProc<1>,  int QueueElements7 = 0, 
//         class Procedure8 = EmptyProc<1>,  int QueueElements8 = 0,  class Procedure9 = EmptyProc<1>,  int QueueElements9 = 0,  class Procedure10 = EmptyProc<1>, int QueueElements10 = 0, class Procedure11 = EmptyProc<1>, int QueueElements11 = 0, 
//         class Procedure12 = EmptyProc<1>, int QueueElements12 = 0, class Procedure13 = EmptyProc<1>, int QueueElements13 = 0, class Procedure14 = EmptyProc<1>, int QueueElements14 = 0, class Procedure15 = EmptyProc<1>, int QueueElements15 = 0>
//struct ElementsMatcher
//{
//  static const int Num = ElementsMatcher<Procedure, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>::Num;
//};
//template<class Procedure,
//         class Procedure0,  class Procedure1, class Procedure2, class Procedure3,
//         class Procedure4, class Procedure5, class Procedure6, class Procedure7,
//         class Procedure8, class Procedure9, class Procedure10, class Procedure11,
//         class Procedure12, class Procedure13, class Procedure14, class Procedure15>
//struct ElementsMatcher<Procedure, Procedure0, 0, Procedure1, 0, Procedure2, 0, Procedure3, 0, Procedure4, 0, Procedure5, 0, Procedure6, 0, Procedure7, 0, Procedure8, 0, Procedure9, 0, Procedure10, 0, Procedure11, 0, Procedure12, 0, Procedure13, 0, Procedure14, 0, Procedure15, 0>
//{
//  static const int Num = 0;
//};
//
//template<class Procedure, 
//         int QueueElements0,  class Procedure1,  int QueueElements1,  class Procedure2,  int QueueElements2,  class Procedure3,  int QueueElements3, 
//         class Procedure4,  int QueueElements4,  class Procedure5,  int QueueElements5,  class Procedure6,  int QueueElements6,  class Procedure7,  int QueueElements7, 
//         class Procedure8,  int QueueElements8,  class Procedure9,  int QueueElements9,  class Procedure10, int QueueElements10, class Procedure11, int QueueElements11, 
//         class Procedure12, int QueueElements12, class Procedure13, int QueueElements13, class Procedure14, int QueueElements14, class Procedure15, int QueueElements15>
//struct ElementsMatcher<Procedure, Procedure, QueueElements0, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,
//         Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,
//         Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,
//         Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>
//{
//  static const int Num = QueueElements0;
//};
//
//
//template<class ProcedureInfo, int MemSize, bool TWarpOptimization,
//         class Procedure0,                 int QElements0,      class Procedure1 = EmptyProc<1>,  int QElements1 = 0,  class Procedure2 = EmptyProc<1>,  int QElements2 = 0,  class Procedure3 = EmptyProc<1>,  int QElements3 = 0, 
//         class Procedure4 = EmptyProc<1>,  int QElements4 = 0,  class Procedure5 = EmptyProc<1>,  int QElements5 = 0,  class Procedure6 = EmptyProc<1>,  int QElements6 = 0,  class Procedure7 = EmptyProc<1>,  int QElements7 = 0, 
//         class Procedure8 = EmptyProc<1>,  int QElements8 = 0,  class Procedure9 = EmptyProc<1>,  int QElements9 = 0,  class Procedure10 = EmptyProc<1>, int QElements10 = 0, class Procedure11 = EmptyProc<1>, int QElements11 = 0, 
//         class Procedure12 = EmptyProc<1>, int QElements12 = 0, class Procedure13 = EmptyProc<1>, int QElements13 = 0, class Procedure14 = EmptyProc<1>, int QElements14 = 0, class Procedure15 = EmptyProc<1>, int QElements15 = 0>
//class SharedDynamicQueue
//{
//#define QueueElementsAdjust(ID) \
//  static const int QueueElements ## ID = QElements ## ID == 0?0:(sizeof(typename Procedure ## ID ::ExpectedData)*QElements ## ID +15)/16*16/sizeof(typename Procedure ## ID::ExpectedData); \
//
//  QueueElementsAdjust(0) QueueElementsAdjust(1) QueueElementsAdjust(2) QueueElementsAdjust(3)
//  QueueElementsAdjust(4) QueueElementsAdjust(5) QueueElementsAdjust(6) QueueElementsAdjust(7)
//  QueueElementsAdjust(8) QueueElementsAdjust(9) QueueElementsAdjust(10) QueueElementsAdjust(11)
//  QueueElementsAdjust(12) QueueElementsAdjust(13) QueueElementsAdjust(14) QueueElementsAdjust(15)
//
//    //dummysize
//  uint dummy[MemSize/4];
//  typedef SharedDynBaseQueue<ProcedureInfo, MemSize, TWarpOptimization> DynQ;
//
//  template<class Proc, int Elements>
//  static std::string name()
//  {
//    if(Proc::ProcedureId < 0)
//      return "";
//    else
//      return std::string("_") + std::to_string((unsigned long long)Proc::ProcedureId) + "(" + std::to_string((unsigned long long)Elements) + ")";
//  }
//
//public:
//  __inline__ __device__ void init()
//  {
//    reinterpret_cast<DynQ*>(this)->init();
//  }
//  __inline__ __device__ void maintain()
//  { 
//    reinterpret_cast<DynQ*>(this)->maintain();
//  }
//  
//  template<class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData data) 
//  { 
//    int maxElements = ElementsMatcher<Procedure, Procedure0, QueueElements0, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>::Num;
//    if(maxElements == 0)
//      return false;
//    return reinterpret_cast<DynQ*>(this)-> template enqueue<Procedure>(data, maxElements);
//  }
//  template<uint ThreadsPerElement, class Procedure>
//  __inline__ __device__ bool enqueue(typename Procedure::ExpectedData* data) 
//  { 
//    int maxElements = ElementsMatcher<Procedure, Procedure0, QueueElements0, Procedure1,QueueElements1,Procedure2,QueueElements2,Procedure3,QueueElements3,Procedure4,QueueElements4,Procedure5,QueueElements5,Procedure6,QueueElements6,Procedure7,QueueElements7,Procedure8,QueueElements8,Procedure9,QueueElements9,Procedure10,QueueElements10,Procedure11,QueueElements11,Procedure12,QueueElements12,Procedure13,QueueElements13,Procedure14,QueueElements14,Procedure15,QueueElements15>::Num;
//    if(maxElements == 0)
//      return false;
//    return reinterpret_cast<DynQ*>(this)-> template enqueue<Procedure, ThreadsPerElement>(data, maxElements);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeue(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { 
//    return reinterpret_cast<DynQ*>(this)->template dequeue<MultiProcedure>(data, procId, maxShared, minPercent);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1, int minPercent = 80)
//  { 
//    return reinterpret_cast<DynQ*>(this)-> template dequeueSelected<MultiProcedure>(data, procId, maxNum, minPercent);
//  }
//  template<bool MultiProcedure>
//  __inline__ __device__ int2 dequeueStartRead(void*& data, int* procId, int maxShared = -1, int minPercent = 80)
//  { 
//    return reinterpret_cast<DynQ*>(this)-> template dequeueStartRead<MultiProcedure>(data, procId, maxShared, minPercent);
//  }
//  
//  template<class Procedure>
//  __inline__ __device__ void finishRead(int id, int num)
//  { 
//    reinterpret_cast<DynQ*>(this)->finishRead(id, num);
//  }
//
//  static std::string name()
//  {
//    return "DynamicShared" +
//      name<Procedure0,QueueElements0>() + name<Procedure1,QueueElements1>() + name<Procedure2,QueueElements2>() + name<Procedure3,QueueElements3>() +
//      name<Procedure4,QueueElements4>() + name<Procedure5,QueueElements5>() + name<Procedure6,QueueElements6>() + name<Procedure7,QueueElements7>() +
//      name<Procedure8,QueueElements8>() + name<Procedure9,QueueElements9>() + name<Procedure10,QueueElements10>() + name<Procedure11,QueueElements11>() +
//      name<Procedure12,QueueElements12>() + name<Procedure13,QueueElements13>() + name<Procedure14,QueueElements14>() + name<Procedure15,QueueElements15>();
//  }
//};


template<class ProcInfo, int MaxSize, class QueueDescription, bool WarpOptimization>
class SharedStaticQueueDirectDefinition : public SharedQueueElement<ProcInfo, typename QueueDescription::Proc, MaxSize, QueueDescription, QueueDescription, WarpOptimization, 0> { };


template<class ProcInfo, int MaxSize, template<typename> class SharedQTraits, bool WarpOptimization>
class SharedStaticQueue : public SharedQueueElement<ProcInfo,
    typename SQElementTraitsPeel<ProcInfo, ProcInfo::NumProcedures-1, 0,SharedQTraits, SharedQTraits<typename Select<ProcInfo,0>::Procedure >::QueueSize>::Proc,
    MaxSize, 
    SQElementTraitsPeel<ProcInfo, ProcInfo::NumProcedures-1, 0,SharedQTraits, SharedQTraits<typename Select<ProcInfo,0>::Procedure >::QueueSize>,
    SQElementTraitsPeel<ProcInfo, ProcInfo::NumProcedures-1, 0,SharedQTraits, SharedQTraits<typename Select<ProcInfo,0>::Procedure >::QueueSize>, WarpOptimization >
{ };


template<int MaxSize, template<typename> class SharedQTraits, bool WarpOptimization>
class SharedStaticQueueTyping
{
  template<class ProcInfo>
  class Type : public SharedQueueElement<ProcInfo,
    typename SQElementTraitsPeel<ProcInfo, ProcInfo::NumProcedures-1, 0,SharedQTraits, SharedQTraits<typename Select<ProcInfo,0>::Procedure >::QueueSize>::Proc,
    MaxSize, 
    SQElementTraitsPeel<ProcInfo, ProcInfo::NumProcedures-1, 0,SharedQTraits, SharedQTraits<typename Select<ProcInfo,0>::Procedure >::QueueSize>,
    SQElementTraitsPeel<ProcInfo, ProcInfo::NumProcedures-1, 0,SharedQTraits, SharedQTraits<typename Select<ProcInfo,0>::Procedure >::QueueSize>, WarpOptimization >
  {};
};
  

template<class ProcedureInfo, template<class ProcedureInfo> class ExternalQueue, template<class ProcedureInfo> class SharedQueue, int SharedQueueFillupThreshold = 80, int GotoGlobalChance = 0>
class  SharedCombinerQueue : protected ExternalQueue<ProcedureInfo>
{
  typedef ExternalQueue<ProcedureInfo> ExtQ;
  typedef SharedQueue<ProcedureInfo> SharedQ;

public:
  static const bool needTripleCall = false;
  static const bool supportReuseInit = ExtQ::supportReuseInit;
  static const int requiredShared = ExtQ::requiredShared + SharedQ :: requiredShared;
  static const int globalMaintainMinThreads = ExtQ::globalMaintainMinThreads;
  static int globalMaintainSharedMemory(int Threads) { return ExtQ::globalMaintainSharedMemory(Threads); }
    

  __inline__ __device__ void init() 
  {
    ExtQ :: init();
  }
  
  template<class PROCEDURE>
  __inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
  {
    return ExtQ :: template enqueueInitial<PROCEDURE>(data);
  }

  template<class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data) 
  {
    extern __shared__ uint s_data[];
    if(GotoGlobalChance == 0 || qrandom::warp_check(100-GotoGlobalChance))
      if(SharedQ :: template enqueue<PROCEDURE>(reinterpret_cast<char*>(s_data), data))
      {
        //printf("went to shared queue\n");
        return true;
      }
    return ExtQ :: template enqueue<PROCEDURE>(data);
  }

  template<int threads, class PROCEDURE>
  __inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
  {
    extern __shared__ uint s_data[];
    if(GotoGlobalChance == 0 || qrandom::warp_check(100-GotoGlobalChance))
      if(SharedQ :: template enqueue<threads, PROCEDURE>(reinterpret_cast<char*>(s_data), data))
      {
        //printf("went to shared queue\n");
        return true;
      }
    return ExtQ :: template enqueue<threads, PROCEDURE>(data);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = -1)
  {
    extern __shared__ uint s_data[];
    int d = SharedQ :: template dequeue<MultiProcedure> (reinterpret_cast<char*>(s_data), data, procId, maxShared, SharedQueueFillupThreshold);
    if(d > 0) return d;
    d = ExtQ :: template dequeue<MultiProcedure>(data, procId, maxShared);
    if(d > 0) return d;
    return  SharedQ :: template dequeue<MultiProcedure> (reinterpret_cast<char*>(s_data), data, procId, maxShared, 0);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
  {
    extern __shared__ uint s_data[];
    int d = SharedQ :: template dequeueSelected<MultiProcedure>(reinterpret_cast<char*>(s_data), data, procId, this->maxShared, SharedQueueFillupThreshold);
    if(d > 0) return d;
    d = ExtQ :: template dequeueSelected<MultiProcedure>(data, procId, this->maxShared);
    if(d > 0) return d;
    return SharedQ :: template dequeueSelected<MultiProcedure>(reinterpret_cast<char*>(s_data), data, procId, this->maxShared, 0);
  }

  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = -1)
  {
    extern __shared__ uint s_data[];
    int2 d = SharedQ:: template dequeueStartRead<MultiProcedure>(reinterpret_cast<char*>(s_data), data, procId, maxShared, SharedQueueFillupThreshold);
    if(d.x > 0)
    {
      procId[1] = d.y | 0x40000000;
      return d.x;
    }
    d.x = ExtQ :: template dequeueStartRead<MultiProcedure>(data, procId, maxShared);
    if(d.x > 0) 
    {
       /*  if(threadIdx.x == 0)
          printf("%d global dequeueStartRead successful %d %d\n", blockIdx.x, d.x, procId[1]);   */ 
        return d.x;
    }
    d = SharedQ :: template dequeueStartRead<MultiProcedure> (reinterpret_cast<char*>(s_data), data, procId, maxShared, 0);
    if(d.x > 0)
    {
      procId[1] = d.y | 0x40000000;
      return d.x;
    }
    return 0;
  }

   template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead1(void*& data, int*& procId, int maxShared = -1)
  {
    extern __shared__ uint s_data[];
    int2 d = SharedQ :: template dequeueStartRead<MultiProcedure>(reinterpret_cast<char*>(s_data), data, procId, maxShared, SharedQueueFillupThreshold);
    procId[1] = d.y | 0x40000000;
    return d.x;
  }
  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead2(void*& data, int*& procId, int maxShared = -1)
  {
    return ExtQ :: template dequeueStartRead<MultiProcedure>(reinterpret_cast<char*>(this->s_data), data, procId, maxShared);
  }
  template<bool MultiProcedure>
  __inline__ __device__ int dequeueStartRead3(void*& data, int*& procId, int maxShared = -1)
  {
    extern __shared__ uint s_data[];
    int2 d = SharedQ:: template dequeueStartRead<MultiProcedure>(reinterpret_cast<char*>(s_data), data, procId, maxShared, 0);
    procId[1] = d.y | 0x40000000;
    return d.x;
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead1(int id,  int num)
  {
    extern __shared__ uint s_data[];
    SharedQ :: template finishRead<PROCEDURE>(reinterpret_cast<char*>(s_data), id & 0x3FFFFFFF, num);
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead2(int id,  int num)
  {
    ExtQ :: template finishRead<PROCEDURE>(id, num);
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead3(int id,  int num)
  {
    finishRead1<PROCEDURE>(id, num);
  }


  template<class PROCEDURE>
  __inline__ __device__ int reserveRead(int maxNum = -1)
  {
    return ExtQ :: template reserveRead <PROCEDURE> (this->data, this->procId, this->maxShared);
  }
  template<class PROCEDURE>
  __inline__ __device__ int startRead(void*& data, int num)
  {
    return  ExtQ :: template startRead<PROCEDURE>(data, num);
  }
  template<class PROCEDURE>
  __inline__ __device__ void finishRead(int id,  int num)
  {
    extern __shared__ uint s_data[];
    if(id & 0x40000000)
    {
      SharedQ :: template finishRead<PROCEDURE>(reinterpret_cast<char*>(s_data), id & 0x3FFFFFFF, num);
      //if(threadIdx.x == 0)
      //printf("%d shared finish read done %d %d\n", blockIdx.x, id,num);
    }
    else
    {
      ExtQ :: template finishRead<PROCEDURE>(id, num);
      // if(threadIdx.x == 0)
      //printf("%d global finish read done %d %d\n", blockIdx.x, id,num);
    }
  }

  __inline__ __device__ void numEntries(int* counts)
  {
    ExtQ :: numEntries(counts);
  }


  __inline__ __device__ void record()
  {
    ExtQ :: record();
  }
  __inline__ __device__ void reset()
  {
    ExtQ :: reset();
  }


  __inline__ __device__ void workerStart()
  { 
    extern __shared__ uint s_data[];
    SharedQ :: init(reinterpret_cast<char*>(s_data));
  }
  __inline__ __device__ void workerMaintain()
  { 
    extern __shared__ uint s_data[];
    SharedQ :: maintain(reinterpret_cast<char*>(s_data));
  }
  __inline__ __device__ void workerEnd()
  { 
    //TODO: what should we do here? enqueue shared elements to global?
  }
  __inline__ __device__ void globalMaintain()
  {
    ExtQ :: globalMaintain();
  }

  static std::string name()
  {
    if(GotoGlobalChance > 0)
      return std::string("SharedCombinedQueue_GolbalProp") + std::to_string((unsigned long long)GotoGlobalChance) + "_" + SharedQ::name() + "/" + ExtQ::name() ;
    return std::string("SharedCombinedQueue_") + SharedQ::name() + "/" + ExtQ::name() ;
  }

};
//
//
//template<template<class ProcedureInfo> class ExternalQueue, template<class ProcedureInfo> class SharedQueue, int SharedQueueFillupThreshold = 80, int GotoGlobalChance = 0>
//struct SharedCombinerQueueTyping
//{
//  template<class ProcedureInfo>
//  class Type : public SharedCombinerQueue<ProcedureInfo, ExternalQueue, SharedQueue, SharedQueueFillupThreshold, GotoGlobalChance> {}; 
//};
//
//
//template<uint ProcedureNumber, uint MemSize, bool TWarpOptimization = true>
//struct SharedSingleSelectedQueueTyping
//{
//  template<class ProcedureInfo>
//  class Type : public SharedSingleSelectedQueue<ProcedureInfo, ProcedureNumber, MemSize, TWarpOptimization> {}; 
//
//};
//
//
//template<uint MemSize, bool TWarpOptimization,
//    int Id0,       int Ratio0,      int Id1 = -1,  int Ratio1 = 0, int Id2  = -1,  int Ratio2 = 0,  int Id3  = -1,  int Ratio3 = 0,
//    int Id4 = -1,  int Ratio4 = 0,  int Id5 = -1,  int Ratio5 = 0, int Id6  = -1,  int Ratio6 = 0,  int Id7  = -1,  int Ratio7 = 0,
//    int Id8 = -1,  int Ratio8 = 0,  int Id9 = -1,  int Ratio9 = 0, int Id10  = -1, int Ratio10 = 0, int Id11  = -1, int Ratio11 = 0,
//    int Id12 = -1, int Ratio12 = 0, int Id13 = -1, int Ratio13 = 0,int Id14  = -1, int Ratio14 = 0, int Id15  = -1, int Ratio15 = 0>
//class SharedMultiQueueTyping
//{
//  static const int SumRatio = Ratio0 + Ratio1 + Ratio2 + Ratio3 + Ratio4 + Ratio5 + Ratio6 + Ratio7 + Ratio8 + Ratio9 + Ratio10 + Ratio11 + Ratio12 + Ratio13 + Ratio14 + Ratio15;
//  static const int Headers = (Ratio0>0?1:0) + (Ratio1>0?1:0) + (Ratio2>0?1:0) + (Ratio3>0?1:0) + (Ratio4>0?1:0) + (Ratio5>0?1:0) + (Ratio6>0?1:0) + (Ratio7>0?1:0) + (Ratio8>0?1:0) + (Ratio9>0?1:0) + (Ratio10>0?1:0) + (Ratio11>0?1:0) + (Ratio12>0?1:0) + (Ratio13>0?1:0) + (Ratio14>0?1:0) + (Ratio15>0?1:0);
//  static const int AvailableMemSize = MemSize - 16*Headers;
//public:
//  template<class ProcedureInfo>
//  class Type : public SharedMultiQueue< ProcedureInfo, MemSize, TWarpOptimization, 
//    typename Select<ProcedureInfo, Id0>::Info,Ratio0*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id0>::Info::ExpectedData),     typename Select<ProcedureInfo, Id1>::Info, Ratio1*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id1>::Info::ExpectedData),    typename Select<ProcedureInfo, Id2>::Info, Ratio2*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id2>::Info::ExpectedData),    typename Select<ProcedureInfo, Id3>::Info, Ratio3*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id3>::Info::ExpectedData),
//    typename Select<ProcedureInfo, Id4>::Info,Ratio4*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id4>::Info::ExpectedData),     typename Select<ProcedureInfo, Id5>::Info, Ratio5*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id5>::Info::ExpectedData),    typename Select<ProcedureInfo, Id6>::Info, Ratio6*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id6>::Info::ExpectedData),    typename Select<ProcedureInfo, Id7>::Info, Ratio7*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id7>::Info::ExpectedData),
//    typename Select<ProcedureInfo, Id8>::Info,Ratio8*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id8>::Info::ExpectedData),     typename Select<ProcedureInfo, Id9>::Info, Ratio9*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id9>::Info::ExpectedData),    typename Select<ProcedureInfo, Id10>::Info, Ratio10*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id10>::Info::ExpectedData), typename Select<ProcedureInfo, Id11>::Info, Ratio11*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id11>::Info::ExpectedData),
//    typename Select<ProcedureInfo, Id12>::Info, Ratio12*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id12>::Info::ExpectedData), typename Select<ProcedureInfo, Id13>::Info, Ratio13*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id13>::Info::ExpectedData), typename Select<ProcedureInfo, Id14>::Info, Ratio14*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id14>::Info::ExpectedData), typename Select<ProcedureInfo, Id15>::Info, Ratio15*AvailableMemSize/SumRatio/sizeof(typename Select<ProcedureInfo, Id15>::Info::ExpectedData)>  
//  {};
//};
//
//template<uint MemSize, bool TWarpOptimization,
//    int Id0,       int Elements0,      int Id1 = -1,  int Elements1 = 0, int Id2  = -1,  int Elements2 = 0,  int Id3  = -1,  int Elements3 = 0,
//    int Id4 = -1,  int Elements4 = 0,  int Id5 = -1,  int Elements5 = 0, int Id6  = -1,  int Elements6 = 0,  int Id7  = -1,  int Elements7 = 0,
//    int Id8 = -1,  int Elements8 = 0,  int Id9 = -1,  int Elements9 = 0, int Id10  = -1, int Elements10 = 0, int Id11  = -1, int Elements11 = 0,
//    int Id12 = -1, int Elements12 = 0, int Id13 = -1, int Elements13 = 0,int Id14  = -1, int Elements14 = 0, int Id15  = -1, int Elements15 = 0>
//class SharedMultiQueueFixedTyping
//{
// public:
//  template<class ProcedureInfo>
//  class Type : public SharedMultiQueue< ProcedureInfo, MemSize, TWarpOptimization, 
//    typename Select<ProcedureInfo, Id0>::Info,Elements0,     typename Select<ProcedureInfo, Id1>::Info, Elements1,    typename Select<ProcedureInfo, Id2>::Info, Elements2,    typename Select<ProcedureInfo, Id3>::Info, Elements3,
//    typename Select<ProcedureInfo, Id4>::Info,Elements4,     typename Select<ProcedureInfo, Id5>::Info, Elements5,    typename Select<ProcedureInfo, Id6>::Info, Elements6,    typename Select<ProcedureInfo, Id7>::Info, Elements7,
//    typename Select<ProcedureInfo, Id8>::Info,Elements8,     typename Select<ProcedureInfo, Id9>::Info, Elements9,    typename Select<ProcedureInfo, Id10>::Info, Elements10, typename Select<ProcedureInfo, Id11>::Info, Elements11,
//    typename Select<ProcedureInfo, Id12>::Info, Elements12, typename Select<ProcedureInfo, Id13>::Info, Elements13, typename Select<ProcedureInfo, Id14>::Info, Elements14, typename Select<ProcedureInfo, Id15>::Info, Elements15>  
//  {};
//};
//
//template<uint TMemSize, bool TWarpOptimization,
//    int Id0,       int Ratio0,      int Id1 = -1,  int Ratio1 = 0, int Id2  = -1,  int Ratio2 = 0,  int Id3  = -1,  int Ratio3 = 0,
//    int Id4 = -1,  int Ratio4 = 0,  int Id5 = -1,  int Ratio5 = 0, int Id6  = -1,  int Ratio6 = 0,  int Id7  = -1,  int Ratio7 = 0,
//    int Id8 = -1,  int Ratio8 = 0,  int Id9 = -1,  int Ratio9 = 0, int Id10  = -1, int Ratio10 = 0, int Id11  = -1, int Ratio11 = 0,
//    int Id12 = -1, int Ratio12 = 0, int Id13 = -1, int Ratio13 = 0,int Id14  = -1, int Ratio14 = 0, int Id15  = -1, int Ratio15 = 0>
//class SharedDynamicTyping
//{
//  static const uint MemSize = TMemSize/16*16;
//public:
//  template<class ProcedureInfo>
//  class Type : public SharedDynamicQueue< ProcedureInfo, MemSize, TWarpOptimization, 
//    typename Select<ProcedureInfo, Id0>::Info,Ratio0*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id0>::Info::ExpectedData),     typename Select<ProcedureInfo, Id1>::Info, Ratio1*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id1>::Info::ExpectedData),    typename Select<ProcedureInfo, Id2>::Info, Ratio2*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id2>::Info::ExpectedData),    typename Select<ProcedureInfo, Id3>::Info, Ratio3*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id3>::Info::ExpectedData),
//    typename Select<ProcedureInfo, Id4>::Info,Ratio4*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id4>::Info::ExpectedData),     typename Select<ProcedureInfo, Id5>::Info, Ratio5*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id5>::Info::ExpectedData),    typename Select<ProcedureInfo, Id6>::Info, Ratio6*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id6>::Info::ExpectedData),    typename Select<ProcedureInfo, Id7>::Info, Ratio7*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id7>::Info::ExpectedData),
//    typename Select<ProcedureInfo, Id8>::Info,Ratio8*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id8>::Info::ExpectedData),     typename Select<ProcedureInfo, Id9>::Info, Ratio9*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id9>::Info::ExpectedData),    typename Select<ProcedureInfo, Id10>::Info, Ratio10*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id10>::Info::ExpectedData), typename Select<ProcedureInfo, Id11>::Info, Ratio11*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id11>::Info::ExpectedData),
//    typename Select<ProcedureInfo, Id12>::Info, Ratio12*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id12>::Info::ExpectedData), typename Select<ProcedureInfo, Id13>::Info, Ratio13*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id13>::Info::ExpectedData), typename Select<ProcedureInfo, Id14>::Info, Ratio14*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id14>::Info::ExpectedData), typename Select<ProcedureInfo, Id15>::Info, Ratio15*(MemSize-16)/100/sizeof(typename Select<ProcedureInfo, Id15>::Info::ExpectedData)>  
//  {};
//};
