#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"

template<uint TQueueSize, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
class QueuePointedStub
{
protected:
  static const int QueueSize = TQueueSize;

  volatile int count;
  uint front, back;

    static std::string name()
  {
      
    if(TWarpOptimization)
      return "PointedWarpoptimized";
    return "Pointed";
  }

  __inline__ __device__ void init() 
  {
    uint lid = threadIdx.x + blockIdx.x*blockDim.x;
    if(lid == 0)
      front = 0, back = 0, count = 0;
  }

  template<int TThreadsPerElement>
  __inline__ __device__  int2 enqueuePrep(int2 last)
  {
    if(TWarpOptimization)
    {
      //combine
      uint mask = __ballot(1);
      int ourcount = __popc(mask)/TThreadsPerElement;
      int mypos = __popc(Softshell::lanemask_lt() & mask);

      int wpos = -1;

      if(mypos == 0)
      {
        int c = atomicAdd(const_cast<int*>(&count), ourcount);
        if(c + ourcount < QueueSize)
          wpos = atomicAdd(&back, ourcount);
        else
        {
          if(TAssertOnOverflow)
          {
            printf("ERROR queue out of elements %d + %d < %d\n",c,ourcount,QueueSize);
            //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
            Softshell::trap();
          }
          atomicSub(const_cast<int*>(&count), ourcount);
        }
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
      if(TThreadsPerElement == 1)
      {
        int c = atomicAdd(const_cast<int*>(&count), 1);
        if(c + 1 < QueueSize)
            return make_int2(atomicAdd(&back, 1), 1);
        else
        {
          if(TAssertOnOverflow)
          {
            printf("ERROR queue out of elements %d + %d < %d\n",c,1,QueueSize);
            //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
            Softshell::trap();
          }
          atomicSub(const_cast<int*>(&count), 1);
          return make_int2(-1,0);
        }
      }
      else
      {
        int pos;
        if(Softshell::laneid() % TThreadsPerElement == 0)
        {
          int c = atomicAdd(const_cast<int*>(&count), 1);
          if(c + 1 < QueueSize)
              pos = atomicAdd(&back, 1);
          else
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements%d + %d < %d\n",c,1,QueueSize);
              //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
              Softshell::trap();
            }
            atomicSub(const_cast<int*>(&count), 1);
            pos = -1;
          }
        }
        pos = warpBroadcast<TThreadsPerElement>(pos, 0);
        if(pos != -1)
          return make_int2(pos,1);
        return make_int2(-1,0);
      }
    }
  }

  template<int TThreadsPerElement>
  __inline__ __device__  void enqueueEnd(int2 pos_ourcount)
  {
  }

  __inline__ __device__ uint2 dequeuePrep(int num)
  { 
    __shared__ uint2 offset_take;
    if(threadIdx.x == 0)
    {
      int c = atomicSub(const_cast<int*>(&count), num);
      if(c < num)
      {
        atomicAdd(const_cast<int*>(&count), min(num,num - c));
        num = max(c, 0);
      }
      offset_take.y = num;
      if(num > 0)
        offset_take.x = atomicAdd(&front, num);
    }
    __syncthreads();
    return offset_take;
  }

  __inline__ __device__ void dequeueEnd(uint2 offset_take)
  {
  }


public:
  __inline__ __device__ int size() const
  {
    return  count;
  }
};


template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
class QueuePointed : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueuePointedStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, AllocStorage<TElementSize, TAdditionalData, TQueueSize, true> >
{
};
  
template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow>
class QueuePointed<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow> : public QueueBuilder<TElementSize, TQueueSize, void, QueuePointedStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, AllocStorage<TElementSize, void, TQueueSize, true> >
{
};


  
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueuePointed_t : public QueuePointed<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueuePointedOpt_t : public QueuePointed<TElementSize, TQueueSize, TAdditionalData, true,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueuePointedNoOverflow_t : public QueuePointed<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueuePointedNoOverflowOpt_t : public QueuePointed<TElementSize, TQueueSize, TAdditionalData, true,false> { };


template<uint TElementSize, class TAdditionalData>
struct AllocatedRingQueueElement
{
  typedef typename StorageElementTyping<TElementSize>::Type ElementData_t;
  typedef typename StorageElementTyping<TElementSize>::Type ElementAdditionalData_t;

  ElementAdditionalData_t elementAdditionalData;
  ElementData_t elementData;
  int size;
    
  template<class T>
  __device__ __inline__ void writeData(T data, TAdditionalData additionalData) volatile
  {
    size = TElementSize;
    elementAdditionalData = *reinterpret_cast<ElementAdditionalData_t*>(&additionalData);
    elementData = *reinterpret_cast<ElementData_t*>(&data);
  }
    
  template<int TThreadsPerElement, class T>
  __device__ __inline__ void writeDataParallel(T* data, TAdditionalData additionalData) volatile
  {
    size = TElementSize;
    multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(&elementData), data);
    multiWrite<TThreadsPerElement, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(&elementAdditionalData), &additionalData);
  }
    
  __device__ __inline__ void readData(void* data, TAdditionalData* additionalData) volatile
  {
    *reinterpret_cast<ElementAdditionalData_t*>(additionalData) = elementAdditionalData;
    readStorageElement(data, &elementData, size);
  }
  __device__ __inline__ uint myRealSize() volatile const
  {
    return sizeof(QueueLinkedHeader) + sizeof(ElementAdditionalData_t) + size;
  }
};

template<uint TElementSize>
struct AllocatedRingQueueElement<TElementSize, void>
{
  typedef typename StorageElementTyping<TElementSize>::Type ElementData_t;
  ElementData_t elementData;
  int size;

  template<class T>
  __device__ __inline__ void writeData(T data) volatile
  {
    size = TElementSize;
    elementData = *reinterpret_cast<ElementData_t*>(&data);
  }
  template<int TThreadsPerElement, class T>
  __device__ __inline__ void writeDataParallel(T* data) volatile
  {
    size = TElementSize;
    multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(&elementData), data);
  }
  __device__ __inline__ void readData(void* data) volatile
  {
    //printf("%d reading %d bytes @ %llx to %llx\n",blockIdx.x, header.size,  &elementData, data);
    readStorageElement(data, &elementData, size);
  }
  __device__ __inline__ uint myRealSize() volatile const
  {
    return sizeof(QueueLinkedHeader) + size;
  }
};



template<template<uint, bool, bool> class Stub, uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization = false,  bool AssertOnOverflow = true> 
class AllocatedRingStubQueue : public Stub<TQueueSize, TWarpOptimization, AssertOnOverflow>,  public BasicQueue<TAdditionalData>, private MemoryAlloc<(TElementSize+sizeof(TAdditionalData))*TQueueSize>
{

public:
  static std::string name()
  {
    return Stub<TQueueSize, TWarpOptimization, AssertOnOverflow>::name();
  }
  __inline__ __device__ void init() 
  {
    Stub<TQueueSize, TWarpOptimization, AssertOnOverflow>::init();
    MemoryAlloc<TElementSize*TQueueSize>::init();
  }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additional) 
    {
      return enqueue<Data>(data, additional);
    }

    template<class Data>
    __device__ bool enqueue(Data data, TAdditionalData additional) 
    {        

      int offset = allocOffset(sizeof(AllocatedRingQueueElement<sizeof(Data),TAdditionalData>));
      if(offset == -1) 
      {
        if(AssertOnOverflow)
        {
          printf("ERROR allocator out of elements: %d \n",offset);
          Softshell::trap();
        }
        return false;
      }

      volatile AllocatedRingQueueElement<sizeof(Data),TAdditionalData>* pdata = reinterpret_cast< volatile AllocatedRingQueueElement<sizeof(Data),TAdditionalData>*>(offsetToPointer(offset));

      pdata->writeData(data, additional);
      __threadfence();

      if(!enqueueOffset(offset))
      {
        freeOffset(offset, sizeof(AllocatedRingQueueElement<sizeof(Data),TAdditionalData>));
        return false;
      }
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data, TAdditionalData add) 
    {  
      int offset = -1;
      if(Softshell::laneid() % Threads == 0)
      {
        offset = allocOffset(sizeof(AllocatedRingQueueElement<sizeof(Data),TAdditionalData>));
      }
      offset = warpBroadcast<Threads>(offset, 0);
      if(offset == -1) 
      {
        if(AssertOnOverflow)
        {
          printf("ERROR allocator out of elements: %d \n",offset);
          Softshell::trap();
        }
        return false;
      }

      volatile AllocatedRingQueueElement<sizeof(Data),TAdditionalData>* pdata = reinterpret_cast< volatile AllocatedRingQueueElement<sizeof(Data),TAdditionalData>*>(offsetToPointer(offset));
      
      pdata->writeDataParallel<Threads>(data, additional);
      __threadfence();
      
      if(Softshell::laneid() % Threads == 0)
      {
         if(!enqueueOffset(offset))
        {
          freeOffset(offset, sizeof(AllocatedRingQueueElement<sizeof(Data),TAdditionalData>));
          offset = -1;
        }
      }
      offset = warpBroadcast<Threads>(offset, 0);
      return offset != -1;
    }

    __inline__ __device__ int dequeue(void* data, TAdditionalData* additionalData, int num)
    {
       __shared__ int take;
      uint* cdata = (uint*)(data);
      if(threadIdx.x == 0)
      {
        take = num;
        for(int i = 0; i < num; ++i)
        {
          int offset = dequeueOffset();
          if(offset < 0)
          {
            take = i;
            i = num;
            break;
          }
          volatile QueueLinkedStoreElement<TElementSize, void>* readdata = reinterpret_cast< volatile AllocatedRingQueueElement<sizeof(Data),TAdditionalData>*>(offsetToPointer(offset));
          readdata->readData(cdata,additionalData+i);
          int s = readdata->size;
          __threadfence();
          free((void*)readdata, readdata->myRealSize());
          cdata += s/sizeof(uint);     
        }
      }
      __syncthreads();
      return take;
    }
};


template<template<uint, bool, bool> class Stub, uint TElementSize, uint TQueueSize,  bool TWarpOptimization, bool AssertOnOverflow> 
class AllocatedRingStubQueue<Stub,TElementSize, TQueueSize, void, TWarpOptimization, AssertOnOverflow> : public Stub<TQueueSize, TWarpOptimization, AssertOnOverflow>,  public BasicQueue<void>, private MemoryAlloc<TElementSize*TQueueSize>
{

public:
  static std::string name()
  {
    return Stub<TQueueSize, TWarpOptimization, AssertOnOverflow>::name();
  }
  __inline__ __device__ void init() 
  {
    Stub<TQueueSize,  TWarpOptimization, AssertOnOverflow>::init();
    MemoryAlloc<TElementSize*TQueueSize>::init();
  }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data) 
    {
      return enqueue<Data>(data);
    }

    template<class Data>
    __device__ bool enqueue(Data data) 
    {        
      //if(threadIdx.x == 0)
      //  printf("%d enqueue\n",blockIdx.x);

      int offset = allocOffset(sizeof(AllocatedRingQueueElement<sizeof(Data),void>));
      if(offset == -1)
      {
        if(AssertOnOverflow)
        {
          printf("ERROR allocator out of elements: %d \n",offset);
          Softshell::trap();
        }
        return false;
      }

      volatile AllocatedRingQueueElement<sizeof(Data),void>* pdata = reinterpret_cast< volatile AllocatedRingQueueElement<sizeof(Data),void>*>(offsetToPointer(offset));

      pdata->writeData(data);
      __threadfence();

      if(!enqueueOffset(offset))
      {
        freeOffset(offset, sizeof(AllocatedRingQueueElement<sizeof(Data),void>));
        //if(threadIdx.x == 0)
        //  printf("%d enqueue failed 2\n",blockIdx.x);
        return false;
      }
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data) 
    {  
      int offset = -1;
      if(Softshell::laneid() % Threads == 0)
      {
        offset = allocOffset(sizeof(AllocatedRingQueueElement<sizeof(Data),void>));
      }
      offset = warpBroadcast<Threads>(offset, 0);
      if(offset == -1) 
      {
        if(AssertOnOverflow)
        {
          printf("ERROR allocator out of elements: %d \n",offset);
          Softshell::trap();
        }
        return false;
      }

      volatile AllocatedRingQueueElement<sizeof(Data),void>* pdata = reinterpret_cast< volatile AllocatedRingQueueElement<sizeof(Data),void>*>(offsetToPointer(offset));
      
      pdata->writeDataParallel<Threads>(data);
      __threadfence();
      
      if(Softshell::laneid() % Threads == 0)
      {
         if(!enqueueOffset(offset))
        {
          freeOffset(offset, sizeof(AllocatedRingQueueElement<sizeof(Data),void>));
          offset = -1;
        }
      }
      offset = warpBroadcast<Threads>(offset, 0);
      return offset != -1;
    }

    __inline__ __device__ int dequeue(void* data, int num)
    {
       __shared__ int take;
      uint* cdata = (uint*)(data);
      if(threadIdx.x == 0)
      {
        take = num;
        for(int i = 0; i < num; ++i)
        {
          int offset = dequeueOffset();
          if(offset < 0)
          {
            take = i;
            i = num;
            break;
          }
          volatile AllocatedRingQueueElement<TElementSize, void>* readdata = reinterpret_cast< volatile AllocatedRingQueueElement<TElementSize,void>*>(offsetToPointer(offset));
          readdata->readData(cdata);
          int s = readdata->size;
          __threadfence();
          free((void*)readdata, readdata->myRealSize());
          cdata += s/sizeof(uint);     
        }
      }
      //if(threadIdx.x == 0)
      //  printf("%d dequeued %d\n",blockIdx.x, take);
      __syncthreads();
      return take;
    }
};




template<uint TQueueSize, bool TWarpOptimization = false, bool TAssertOnOverflow = true>
class ShannHuangChenStub
{
protected:

  

  static const int QueueSize = TQueueSize;

 // volatile int count;

  volatile int p[QueueSize];

  volatile uint front;
  volatile uint back;

  static std::string name()
  {
    return std::string("ShannHuangChen") + (TWarpOptimization?"Warpoptimized":"");
  }

  __inline__ __device__ void init() 
  {
    uint lid = threadIdx.x + blockIdx.x*blockDim.x;
    if(lid == 0)
      front = 0, back = 0;
    for(int i = lid; i < QueueSize; i+=blockDim.x*gridDim.x)
      p[i] = -1;
  }

  __inline__ __device__ bool enqueueOffset(int offset)
  {
    while(true)
    {
      uint b = back;
      uint pos = b;
      bool last = true;
      if(TWarpOptimization)
      {
        uint mask = __ballot(1);
        int mypos = __popc(Softshell::lanemask_lt() & mask);
        int src = __ffs(mask)-1;
        pos = warpBroadcast<32>(pos, src) + mypos;
        last = __popc(Softshell::lanemask_gt() & __ballot(1)) == 0;
      }

      int x = p[pos % QueueSize];
      __threadfence();
      if(b == back)
      {
        if(pos >= front + QueueSize)
        {
          if(TAssertOnOverflow)
          {
            printf("ERROR queue out of elements %d %d - %d\n",front,b,QueueSize);
            Softshell::trap();
          }
          return false;
        }
        if(x < 0)
        {
          if(atomicCAS(const_cast<int*>(p + (b % QueueSize)), x, offset) == x)
          {
            __threadfence();
            //if(TWarpOptimization)
            //  last = __popc(Softshell::lanemask_gt() & __ballot(1)) == 0;
            if(last)
              atomicCAS(const_cast<uint*>(&back), b, pos+1);
            return true;
          }
        }
        else
        {
          atomicCAS(const_cast<uint*>(&back), b, pos+1);
          __threadfence();
        }
      }
    }
  }

   __inline__ __device__ int dequeueOffset()
  {
    while(true)
    {
      int f = front;
      int x = p[f % QueueSize];
      __threadfence();
      if(f == front)
      {
        if(f >= back)
          return -1;
        if(x >= 0)
        {
          if(atomicCAS(const_cast<int*>(p + (f % QueueSize)), x, -1) == x)
          {
            __threadfence();
            atomicCAS(const_cast<uint*>(&front), f, f+1u);
            return x;
          }
        }
        else
        {
           atomicCAS(const_cast<uint*>(&front), f, f+1u);
           __threadfence();
        }
      }
    }
  }
 


public:
  __inline__ __device__ int size() const
  {
    int diff = back - front;
    return  diff < 0 ? 0 : diff > QueueSize ? QueueSize : diff;
  }
};


template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool WarpOptimization = false, bool AssertOnOverflow = true> 
class QueueShannHuangChen : public AllocatedRingStubQueue<ShannHuangChenStub, TElementSize, TQueueSize, TAdditionalData, WarpOptimization, AssertOnOverflow> { };




template<uint TQueueSize, bool TWarpOptimization = false,  bool TAssertOnOverflow = true, int TLazyUpdate = 2>
class TsigasZhangStub
{
protected:

  static const int LazyUpdate = TLazyUpdate;
  static const int QueueSize = TQueueSize;

 // volatile int count;

  volatile int p[QueueSize];

  volatile uint front;
  volatile uint back;
  int vnull;

  static const int null0 = -1;
  static const int null1 = -2;

  static std::string name()
  {
    return std::string("TsigasZhang") + (TWarpOptimization?"Warpoptimized":"") + std::to_string((unsigned long long)TLazyUpdate);
  }

  __inline__ __device__ void init() 
  {
    uint lid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int i = lid; i < QueueSize; i+=blockDim.x*gridDim.x)
      p[i] = null0;
    if(lid == 0)
    {
      front = 0, back = 1;
      vnull = null1;
      p[0] = null1;
    }
  }

  __inline__ __device__ bool enqueueOffset(int offset)
  {
    while(true)
    {
      int mypos = 0;
      uint mask = 1;
      if(TWarpOptimization)
      {
        mask = __ballot(1);
        mypos = __popc(Softshell::lanemask_lt() & mask);
      }

      uint abe = 0;
      int tt = -1;
      int be = back;
      if(mypos == 0)
      {
        abe = be;
        tt = p[abe];
        uint temp = (abe + 1) % QueueSize;
        __threadfence();
        //printf("%d %d, start search @ %d\n",blockIdx.x, threadIdx.x, abe);
        while(tt != null0 && tt != null1)
        {
          //printf("%d %d, search increased %d->%d, because values is %d\n",blockIdx.x, threadIdx.x, abe, abe+1, tt);
          if(be != back) 
            break;
          if(temp == front)
            break;
          tt = p[temp];
          abe = temp;
          temp = (abe + 1) % QueueSize;
          __threadfence();
         
        }
        __threadfence();
        if(be != back) 
        {
          be = -1;
        }
        else
        {
          if(temp == front)
          {
            abe = (temp + 1) % QueueSize;
            tt = p[abe];
            if(tt != null0 && tt != null1)
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements %d %d - %d\n",front,be,QueueSize);
                Softshell::trap();
              }
              be = -2;
            }
            else
            {
              if(!abe)
                vnull = tt;
              __threadfence();
              atomicCAS(const_cast<uint*>(&front), temp, abe);
              __threadfence();
              be = -1;
            }
          }
          __threadfence();
        }
      }

      if(be != back) 
        continue;

      if(TWarpOptimization)
      {
        int src = __ffs(mask)-1;
        be = warpBroadcast<32>(be, src);
        abe = (warpBroadcast<32>(abe, src) + mypos) % QueueSize;
        tt = warpBroadcast<32>(tt, src);
      }

      if(be == -2)
        return false;

      if(be >= 0)
      {
        if(atomicCAS(const_cast<int*>(p + abe), tt, offset) == tt)
        {
          __threadfence();
          //printf("%d %d, inserted @ %d val %d->%d, setting back(%d) to %d->%d\n",blockIdx.x, threadIdx.x, abe, tt, offset, (temp % LazyUpdate == 0), be, temp);
          if(!TWarpOptimization || ((__popc(mask) - mypos) <=  LazyUpdate) )
          if((abe+1) % LazyUpdate == 0)
            atomicCAS(const_cast<uint*>(&back), be, (abe+1)% QueueSize);
          return true;
        }
      }
    }
  }

   __inline__ __device__ int dequeueOffset()
  {
    while(true)
    {
      int f = front;
      int temp = (f + 1) % QueueSize;
      int tt = p[temp];
      __threadfence();
      while(tt == null0 ||tt == null1)
      {
        if(f != front)
          break;
        if(temp == back)
          return -1;
        temp = (temp + 1) % QueueSize;
        tt = p[temp];
        __threadfence();
      }
      __threadfence();
      if(f != front)
        continue;
      if(temp == back)
      {
        atomicCAS(const_cast<uint*>(&back), temp, (temp+1u)% QueueSize);
        continue;
      }
      int tnull;
      if(temp)
      {
        if(temp < f)
          tnull = p[0];
        else
          tnull = vnull;
      }
      else
        tnull = vnull == null0 ? null1 : null0;
      __threadfence();
      if(f != front)
        continue;

      if(atomicCAS(const_cast<int*>(p + temp), tt, tnull) == tt)
      {
        if(!temp) vnull = tnull;
        if(temp % LazyUpdate == 0)
          atomicCAS(const_cast<uint*>(&front), f, temp);
        return tt;
      }
    }
  }
 


public:
  __inline__ __device__ int size() const
  {
    int b = back;
    int f = front;

    int diff = b - f;
    return  diff < LazyUpdate ? 0 : diff > QueueSize ? QueueSize : diff;
  }
};

template<uint TQueueSize, bool TWarpOptimization = false, bool TAssertOnOverflow = true>
class TsigasZhangStub2 : public TsigasZhangStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, 2> { };

template<uint TQueueSize, bool TWarpOptimization = false, bool TAssertOnOverflow = true>
class TsigasZhangStub4 : public TsigasZhangStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, 4> { };

template<uint TQueueSize, bool TWarpOptimization = false, bool TAssertOnOverflow = true>
class TsigasZhangStub8 : public TsigasZhangStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, 8> { };


template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool WarpOptimization = false, bool AssertOnOverflow = true> 
class QueueTsigasZhang : public AllocatedRingStubQueue< TsigasZhangStub2, TElementSize, TQueueSize, TAdditionalData, WarpOptimization, AssertOnOverflow> { };

template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool WarpOptimization = false, bool AssertOnOverflow = true> 
class QueueTsigasZhang4 : public AllocatedRingStubQueue< TsigasZhangStub4, TElementSize, TQueueSize, TAdditionalData, WarpOptimization, AssertOnOverflow> { };

template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool WarpOptimization = false, bool AssertOnOverflow = true> 
class QueueTsigasZhang8 : public AllocatedRingStubQueue< TsigasZhangStub8, TElementSize, TQueueSize, TAdditionalData, WarpOptimization, AssertOnOverflow> { };
