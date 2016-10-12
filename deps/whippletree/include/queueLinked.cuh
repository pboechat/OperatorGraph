#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"


  struct QueueLinkedHeader
  {
    int next;
    int size;
  };

  template<uint TElementSize, class TAdditionalData>
  struct QueueLinkedStoreElement
  {
    typedef typename StorageElementTyping<TElementSize>::Type ElementData_t;
    typedef typename StorageElementTyping<TElementSize>::Type ElementAdditionalData_t;

    QueueLinkedHeader header;
    ElementAdditionalData_t elementAdditionalData;
    ElementData_t elementData;
    
    template<class T>
    __device__ __inline__ void writeData(T data, TAdditionalData additionalData) volatile
    {
      header.size = TElementSize;
      elementAdditionalData = *reinterpret_cast<ElementAdditionalData_t*>(&additionalData);
      elementData = *reinterpret_cast<ElementData_t*>(&data);
    }
    
    template<int TThreadsPerElement, class T>
    __device__ __inline__ void writeDataParallel(T* data, TAdditionalData additionalData) volatile
    {
      header.size = TElementSize;
      multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(&elementData), data);
      multiWrite<TThreadsPerElement, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(&elementAdditionalData), &additionalData);
    }
    
    __device__ __inline__ void readData(void* data, TAdditionalData* additionalData) volatile
    {
      *reinterpret_cast<ElementAdditionalData_t*>(additionalData) = elementAdditionalData;
      readStorageElement(data, &elementData, header.size);
    }
    __device__ __inline__ uint myRealSize() volatile const
    {
      return sizeof(QueueLinkedHeader) + sizeof(ElementAdditionalData_t) + header.size;
    }
  };

  template<uint TElementSize>
  struct QueueLinkedStoreElement<TElementSize, void>
  {
    typedef typename StorageElementTyping<TElementSize>::Type ElementData_t;
    QueueLinkedHeader header;
    ElementData_t elementData;

    template<class T>
    __device__ __inline__ void writeData(T data) volatile
    {
      header.size = TElementSize;
      elementData = *reinterpret_cast<ElementData_t*>(&data);
    }
    template<int TThreadsPerElement, class T>
    __device__ __inline__ void writeDataParallel(T* data) volatile
    {
      header.size = TElementSize;
      multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(&elementData), data);
    }
    __device__ __inline__ void readData(void* data) volatile
    {
      //printf("%d reading %d bytes @ %llx to %llx\n",blockIdx.x, header.size,  &elementData, data);
      readStorageElement(data, &elementData, header.size);
    }
    __device__ __inline__ uint myRealSize() volatile const
    {
      return sizeof(QueueLinkedHeader) + header.size;
    }
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization,  template<uint MemSize> class MemAlloc = MemoryAllocFastest, int MemMultiplier = 1>
  class QueueLinkedBase : protected MemAlloc<sizeof(QueueLinkedStoreElement<TElementSize, TAdditionalData>)*TQueueSize*MemMultiplier>
  {
  protected:
    typedef QueueLinkedStoreElement<TElementSize, TAdditionalData> LinkedElement;

    static const int QueueSize = TQueueSize;

    volatile int front;
    volatile int back;
    volatile int count;

    template<uint TThreadsPerElement, class Data>
    __inline__ __device__ volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* prepareEnqueue(int& offset)
    {
      if(TThreadsPerElement == 1)
      {
        int elements = atomicAdd(const_cast<int*>(&count), 1);
        if(elements >= QueueSize)
        {
          atomicSub(const_cast<int*>(&count),1);
          return nullptr;
        }

        offset = allocOffset(sizeof(QueueLinkedStoreElement<sizeof(Data), TAdditionalData>));
        return reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(offset));
      }
      else
      {
        offset = -1;
        if(Softshell::laneid() % TThreadsPerElement == 0)
        {
          int elements = atomicAdd(const_cast<int*>(&count), 1);
          if(elements >= QueueSize)
            atomicSub(const_cast<int*>(&count),1);
          else
            offset = allocOffset(sizeof(QueueLinkedStoreElement<sizeof(Data), TAdditionalData>));
        }
        offset = warpBroadcast<TThreadsPerElement>(offset, 0);
        //offset = __shfl(offset, 0, TThreadsPerElement);
        if(offset == -1)
          return nullptr;
        else
          return reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(offset));
      }
    }

    template<uint TThreadsPerElement, class Data>
    __inline__ __device__  void finishEnqueue(int offset, volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata)
    {
      
      int final = offset;

      if(TWarpOptimization && (TThreadsPerElement == 1 || Softshell::laneid()%TThreadsPerElement == 0))
      {
        //distribute pointers
        int mnext;
        uint mask = __ballot(1);
        //find id of next thread
        int next_id = __ffs(mask & Softshell::lanemask_gt()) -1;
        //get next's offsets
        mnext = warpShfl<32>(offset, next_id);
        if(next_id == -1) // i am the last, so set to -1
          mnext = -1;
        pdata->header.next = mnext;
        __threadfence();

        //get id of the last one
        int last = 31 - __clz(mask);
        //get offset of the last one
        final = warpBroadcast<32>(offset, last);
        //all but the leading one are done
        if(__popc(Softshell::lanemask_lt() & mask) != 0)
          return;
      }

      if(TThreadsPerElement == 1 || Softshell::laneid()%TThreadsPerElement == 0)
      {
        if(!TWarpOptimization)
        {
          pdata->header.next = -1;
          __threadfence();
        }

      
        int pos = atomicExch(const_cast<int*>(&back), final);

        if(pos != -1)
        {
          volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* prevdata = reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(pos));
          prevdata->header.next = offset;
        }
        else
          while(atomicCAS(const_cast<int*>(&front), -1, static_cast<int>(offset)) != -1)
            __threadfence();
      }
    }


    __inline__ __device__ int dequeueOne()
    {
      int f = front;
      if(f == -1) 
        return -1;

      int n = reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(f))->header.next;

      while(true)
      {
        if(n == -1)
        {
          int oldb = atomicCAS(const_cast<int*>(&back), f, -1);
          if(oldb == f)
          {
            atomicExch(const_cast<int*>(&front), -1);
            return f;
          }
          else
          {
            f = front;
            if(f == -1) 
              return -1;
            n = reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(f))->header.next;
          }
        }
        else
        {
          int fnew = atomicCAS(const_cast<int*>(&front), f, n);
          if(fnew == -1) 
            return -1;
          if(fnew != f)
            f = fnew, 
            n = reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(fnew))->header.next;
          else
            return f;
        }
      }
    }


  public:
    static std::string name()
    {
      return std::string("Linked") + (TWarpOptimization?"Warpoptimized":"");
    }

    __inline__ __device__ void init() 
    {
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
        count = 0, front = -1, back = -1;
      MemAlloc<sizeof(QueueLinkedStoreElement<TElementSize, TAdditionalData>)*TQueueSize*MemMultiplier>::init();
    }

    __inline__ __device__ uint size() const
    {
      return count;
    }
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization, template<uint MemSize> class MemAlloc = MemoryAllocFastest, int MemMultiplier = 1>
  class QueueLinkedImpl : public QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>, public ::BasicQueue<TAdditionalData>
  {

  public:
    static std::string name()
    {
      return QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>::name();
    }

    __inline__ __device__ void init() 
    {
      QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>::init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additionalData) 
    {
      return enqueue<Data>(data, additionalData);
    }

    template<class Data>
    __device__ bool enqueue(Data data, TAdditionalData additionalData) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata = 
        QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>:: template prepareEnqueue<1, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeData(data, additionalData);
      
      QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>:: template finishEnqueue<1,Data>(offset, pdata);
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data, TAdditionalData additionalData) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata = 
        QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>:: template prepareEnqueue<Threads, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeDataParallel<Threads>(data, additionalData);
      
      QueueLinkedBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc, MemMultiplier>:: template finishEnqueue<Threads,Data>(offset, pdata);
      return true;
    }

    __inline__ __device__ int dequeue(void* data, TAdditionalData* additionalData, int num)
    {
      __shared__ int take;
      uint* cdata = (uint*)(data);
      if(threadIdx.x == 0)
      {
        int c = atomicSub(const_cast<int*>(&count), num);
        if(c < num)
        {
          atomicAdd(const_cast<int*>(&count), min(num,num - c));
          num = max(c, 0);
        }
        take = num;
        for(int i = 0; i < num; ++i)
        {
          int offset = dequeueOne();
          if(offset == -1)
          {
            atomicAdd(const_cast<int*>(&count), num-i);
            take = i;
            break;
          }
          volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>* readdata =
            reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(offset));
          readdata->readData(cdata, additionalData + i);
          cdata += readdata->header.size;
          freeOffset(offset,readdata->myRealSize());
        }
      }
      __syncthreads();
      return take;
    }
  };

  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, template<uint > class MemAlloc, int MemMultiplier>
  class QueueLinkedImpl<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier> : public QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>, public ::BasicQueue<void>
  {

  public:
    static std::string name()
    {
      return QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>::name();
    }

    __inline__ __device__ void init() 
    {
      QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>::init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data) 
    {
      return enqueue<Data>(data);
    }

    template<class Data>
    __device__ bool enqueue(Data data) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), void>* pdata = 
        QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>:: template prepareEnqueue<1,Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeData(data);
      
      QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>:: template finishEnqueue<1,Data>(offset, pdata);
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), void>* pdata = 
        QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>:: template prepareEnqueue<Threads, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeDataParallel(data);
      
      QueueLinkedBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc, MemMultiplier>:: template finishEnqueue<Threads,Data>(offset, pdata);
      return true;
    }

    __inline__ __device__ int dequeue(void* data, int num)
    {
      __shared__ int take;
      uint* cdata = (uint*)(data);
      if(threadIdx.x == 0)
      {
        int c = atomicSub(const_cast<int*>(&count), num);
        if(c < num)
        {
          atomicAdd(const_cast<int*>(&count), min(num,num - c));
          num = max(c, 0);
        }
        take = num;
        for(int i = 0; i < num; ++i)
        {
          int offset = dequeueOne();
          if(offset == -1)
          {
            atomicAdd(const_cast<int*>(&count), num-i);
            take = i;
            break;
          }
          volatile QueueLinkedStoreElement<TElementSize, void>* readdata = reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, void>*>(offsetToPointer(offset));
          readdata->readData(cdata);
          cdata += readdata->header.size/sizeof(uint);
          freeOffset(offset,readdata->myRealSize());
        }
      }
      __syncthreads();
      return take;
    }
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinked : public QueueLinkedImpl<TElementSize, TQueueSize, TAdditionalData, false, MemoryAllocFastest, 2>
  {
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedRealistic : public QueueLinkedImpl<TElementSize, TQueueSize, TAdditionalData, false, MemoryAlloc, 2>
  {
  public:
    static std::string name()
    {
      return "LinkedRealistic";
    }
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedOpt : public QueueLinkedImpl<TElementSize, TQueueSize, TAdditionalData, true, MemoryAllocFastest, 2>
  {
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedOptNoOverflow : public QueueLinkedImpl<TElementSize, TQueueSize, TAdditionalData, false, MemoryAllocFastest, 2>
  {
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedRealisticOpt  : public QueueLinkedImpl<TElementSize, TQueueSize, TAdditionalData, true, MemoryAlloc, 2>
  {
  public:
    static std::string name()
    {
      return "LinkedRealisticWarpoptimized";
    }
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedRealisticOptNoOverflow  : public QueueLinkedImpl<TElementSize, TQueueSize, TAdditionalData, false, MemoryAlloc, 2>
  {
  public:
    static std::string name()
    {
      return "LinkedRealisticWarpoptimized";
    }
  };

  /**********************************************************************************************/
  //                              MICHAEL SCOTT QUEUE                                           //
  /**********************************************************************************************/





  template<uint TElementSize, uint TQueueSize, class TAdditionalData,  bool TWarpOptimization, template<uint MemSize> class MemAlloc = MemoryAllocFastest>
  class QueueLinkedMichaelScottBase : protected MemAlloc<sizeof(QueueLinkedStoreElement<TElementSize, TAdditionalData>)*TQueueSize>
  {
  protected:
    typedef QueueLinkedStoreElement<TElementSize, TAdditionalData> LinkedElement;

    static const int QueueSize = TQueueSize;

    volatile int front;
    volatile int back;
    volatile int count;

    template<uint TThreadsPerElement, class Data>
    __inline__ __device__ volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* prepareEnqueue(int& offset)
    {
      if(TThreadsPerElement == 1)
      {
        int elements = atomicAdd(const_cast<int*>(&count), 1);
        if(elements >= QueueSize)
        {
          atomicSub(const_cast<int*>(&count),1);
          return nullptr;
        }

        offset = allocOffset(sizeof(QueueLinkedStoreElement<sizeof(Data), TAdditionalData>));
        if(offset == -1)
          return nullptr;
        return reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(offset));
      }
      else
      {
        offset = -1;
        if(Softshell::laneid() % TThreadsPerElement == 0)
        {
          int elements = atomicAdd(const_cast<int*>(&count), 1);
          if(elements >= QueueSize)
            atomicSub(const_cast<int*>(&count),1);
          else
            offset = allocOffset(sizeof(QueueLinkedStoreElement<sizeof(Data), TAdditionalData>));
        }
        offset = warpBroadcast<TThreadsPerElement>(offset, 0);
        //offset = __shfl(offset, 0, TThreadsPerElement);
        if(offset == -1)
          return nullptr;
        else
          return reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(offset));
      }
    }

    template<uint TThreadsPerElement, class Data>
    __inline__ __device__  void finishEnqueue(int offset, volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata)
    {

      int final = offset;
      if(TWarpOptimization && (TThreadsPerElement == 1 || Softshell::laneid()%TThreadsPerElement == 0))
      {
        //distribute pointers
        int mnext;
        uint mask = __ballot(1);
        //find id of next thread
        int next_id = __ffs(mask & Softshell::lanemask_gt()) -1;
        //get next's offsets
        mnext = warpShfl<32>(offset, next_id);
        if(next_id == -1) // i am the last, so set to -1
          mnext = -1;
        pdata->header.next = mnext;
        __threadfence();

        //get id of the last one
        int last = 31 - __clz(mask);
        //get offset of the last one
        final = warpBroadcast<32>(offset, last);
        //all but the leading one are done
        if(__popc(Softshell::lanemask_lt() & mask) != 0)
          return;
      }

      if(TThreadsPerElement == 1 || Softshell::laneid()%TThreadsPerElement == 0)
      {
         if(!TWarpOptimization)
        {
          pdata->header.next = -1;
          __threadfence();
        }

        int b = -1;
        while(true)
        {
          b = back;
          int n = reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(b))->header.next;
          if(b == back)
          {
            if(n == -1)
            {
              if(atomicCAS(const_cast<int*>(&reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(b))->header.next), n, offset) == n)
                break;
            }
            else
              atomicCAS(const_cast<int*>(&back), b, n);
          }
        }
        atomicCAS(const_cast<int*>(&back), b, final);
      }
    }

    __inline__ __device__ bool startDequeue()
    {
      if(count <= 0)
        return false;
      if(atomicSub(const_cast<int*>(&count),1) <= 0)
      {
        atomicAdd(const_cast<int*>(&count),1);
        return false;
      }
      return true;
    }
    __inline__ __device__ void failDequeue()
    {
      atomicAdd(const_cast<int*>(&count),1);
    }

    __inline__ __device__ volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>* prepareDequeue(int& f, int& n)
    {
      n = -1;
      


      while(true)
      {
        f = front;
        int b = back;
        n = reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(f))->header.next;
        if(f == front)
        {
          if(f == b)
          {
            if(n == -1)
              return nullptr;
            atomicCAS(const_cast<int*>(&back), b, n);
          }
          else
          {
            return reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(n));
          }
        }
      }
    }

    __inline__ __device__ bool finishDequeue(int f, int n)
    {
      if(atomicCAS(const_cast<int*>(&front),f, n)==f)
      {
        //printf("%d freeing %d \n",blockIdx.x, f);
        freeOffset(f,reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(f))->myRealSize());
        return true;
      }
      //printf("%d cas failed on %d \n",blockIdx.x, f);
      return false;
    }

  public:
    static std::string name()
    {
      return "MichaelScott";
    }

    __inline__ __device__ void init() 
    {
      MemAlloc<sizeof(QueueLinkedStoreElement<TElementSize, TAdditionalData>)*TQueueSize>::init();
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
      {
        int offset = allocOffset(sizeof(QueueLinkedStoreElement<TElementSize, TAdditionalData>));
        reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(TElementSize), TAdditionalData>*>(offsetToPointer(offset))->header.next = -1;
        front = offset, back = offset, count = 0;
      }
      
    }

    __inline__ __device__ int size()
    {
      return count;
      /*if(front == back)
        return 0;
      return 1024;*/
    }
  };

    

  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization, template<uint > class MemAlloc = MemoryAllocFastest>
  class QueueLinkedMichaelScottImp : public QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>, public ::BasicQueue<TAdditionalData>
  {

  public:
    static std::string name()
    {
      return QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>::name();
    }

    __inline__ __device__ void init() 
    {
      QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>::init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additionalData) 
    {
      return enqueue<Data>(data, additionalData);
    }

    template<class Data>
    __device__ bool enqueue(Data data, TAdditionalData additionalData) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata = 
        QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template prepareEnqueue<1, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeData(data, additionalData);
      
      QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template finishEnqueue<1,Data>(offset, pdata);
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data, TAdditionalData additionalData) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata = 
        QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template prepareEnqueue<Threads, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeDataParallel<Threads>(data, additionalData);
      
      QueueLinkedMichaelScottBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template finishEnqueue<Threads,Data>(offset, pdata);
      return true;
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
          if(!startDequeue())
          {
            take = i;
            i = num;
            break;
          }

          int f, n;
          while(true)
          {
            volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>* readdata = prepareDequeue(f, n);
            if(n == -1)
            {
              failDequeue();
              take = i;
              i = num;
              break;
            }
            int s = readdata->header.size;
            readdata->readData(cdata, additionalData + i);
            __threadfence();
            if(finishDequeue(f, n))
            {
              cdata += s/sizeof(uint);
              break;
            }
          }
        }
      }
      __syncthreads();
      return take;
    }
  };
  

  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization,  template<uint > class MemAlloc>
  class QueueLinkedMichaelScottImp<TElementSize,TQueueSize,void,TWarpOptimization,MemAlloc> : public QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>, public ::BasicQueue<void>
  {

  public:
    static std::string name()
    {
      return QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>::name();
    }

    __inline__ __device__ void init() 
    {
      QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>::init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data) 
    {
      return enqueue<Data>(data);
    }

    template<class Data>
    __device__ bool enqueue(Data data) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), void>* pdata = 
        QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template prepareEnqueue<1, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeData(data);
      __threadfence();
      
      QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template finishEnqueue<1,Data>(offset, pdata);
      //if(threadIdx.x%32 == 0)
      //  printf("%d %d enqueue succeeded\n",blockIdx.x, threadIdx.x);
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), void>* pdata = 
        QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template prepareEnqueue<Threads, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeDataParallel<Threads>(data);
      __threadfence();
      
      QueueLinkedMichaelScottBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template finishEnqueue<Threads,Data>(offset, pdata);
      return true;
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
          //printf("%d start dequeue %d\n",blockIdx.x,i);
          if(!startDequeue())
          {
            take = i;
            i = num;
            //printf("%d dequeue out 0\n",blockIdx.x);
            break;
          }
          int f, n;
          while(true)
          {
            volatile QueueLinkedStoreElement<TElementSize, void>* readdata = prepareDequeue(f, n);
            if(n == -1)
            {
              //printf("%d dequeue out 1\n",blockIdx.x);
              failDequeue();
              take = i;
              i = num;
              break;
            }
            //printf("%d read data %llx\n",blockIdx.x,readdata);
            readdata->readData(cdata);
            int s = readdata->header.size;
            __threadfence();
            if(finishDequeue(f, n))
            {
              //printf("(%d read data %llx (%d) succeeded\n",blockIdx.x,readdata,i);
              cdata += s/sizeof(uint);
              break;
            }
            //else
             // printf("%d could not read: %llx (%d)\n",blockIdx.x, readdata,i);
          }
        }
      }
      __syncthreads();
      //if(threadIdx.x == 1)
      //  printf("%d dequeue succeeded with: %d\n",blockIdx.x, take);
      return take;
    }
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedMichaelScottFastAlloc : public QueueLinkedMichaelScottImp<TElementSize, TQueueSize, TAdditionalData, false, MemoryAllocFastest>
  {
  public:
    static std::string name()
    {
      return "MichaelScottFast";
    }
  };
  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedMichaelScottOptFastAlloc : public QueueLinkedMichaelScottImp<TElementSize, TQueueSize, TAdditionalData, true, MemoryAllocFastest>
  {
  public:
    static std::string name()
    {
      return "MichaelScottFastWarpoptimized";
    }
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedMichaelScott : public QueueLinkedMichaelScottImp<TElementSize, TQueueSize, TAdditionalData, false, MemoryAlloc>
  {
  public:
    static std::string name()
    {
      return "MichaelScott";
    }
  };

    template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedMichaelScottOpt : public QueueLinkedMichaelScottImp<TElementSize, TQueueSize, TAdditionalData, true, MemoryAlloc>
  {
  public:
    static std::string name()
    {
      return "MichaelScottWarpoptimized";
    }
  };

  /**********************************************************************************************/
  //                                      BASKET QUEUE                                           //
  /**********************************************************************************************/



  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization, template<uint MemSize> class MemAlloc = MemoryAllocFastest>
  class QueueLinkedBasketBase : public QueueLinkedMichaelScottBase<TElementSize,TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc> 
  {
  protected:
    typedef QueueLinkedStoreElement<TElementSize, TAdditionalData> LinkedElement;

    static const int QueueSize = TQueueSize;

    static const int Deleted = 0x40000000;
    static const int MaxHops = 2;

    __inline__ __device__ volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>* toPointer(int offset)
    {
      if(offset == -1)
        return nullptr;
      return reinterpret_cast<volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>*>(offsetToPointer(offset & (~Deleted)));
    }
    __inline__ __device__  bool isNull(int offset)
    {
      return offset == -1;
    }
    __inline__ __device__  bool isDeleted(int offset)
    {
      return (offset & Deleted) == Deleted;
    }
    __inline__ __device__ int cleanOffset(int offset)
    {
      return offset & (~Deleted);
    }
    __inline__ __device__ bool equal(int offset0, int offset1)
    {
      return cleanOffset(offset0) == cleanOffset(offset1);
    }



    template<uint TThreadsPerElement, class Data>
    __inline__ __device__  void finishEnqueue(int offset, volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata)
    {
      int final = offset;
      if(TWarpOptimization && (TThreadsPerElement == 1 || Softshell::laneid()%TThreadsPerElement == 0))
      {
        //distribute pointers
        int mnext;
        uint mask = __ballot(1);
        //find id of next thread
        int next_id = __ffs(mask & Softshell::lanemask_gt()) -1;
        //get next's offsets
        mnext = warpShfl<32>(offset, next_id);
        if(next_id == -1) // i am the last, so set to -1
          mnext = -1;
        pdata->header.next = mnext;
        __threadfence();

        //get id of the last one
        int last = 31 - __clz(mask);
        //get offset of the last one
        final = warpBroadcast<32>(offset, last);
        //all but the leading one are done
        if(__popc(Softshell::lanemask_lt() & mask) != 0)
          return;
        pdata = reinterpret_cast<volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>*>(offsetToPointer(final));
      }

      if(TThreadsPerElement == 1 || Softshell::laneid()%TThreadsPerElement == 0)
      {
        if(!TWarpOptimization)
        {
          pdata->header.next = -1;
          __threadfence();
        }

        int b = -1;
        while(true)
        {
          b = back;
          int n = toPointer(b)->header.next;
          if( equal(b,back) )
          {
            if(isNull(n))
            {
              pdata->header.next = -1;
              __threadfence();
              if(atomicCAS(const_cast<int*>(&toPointer(b)->header.next), n, offset) == n)
              {
                atomicCAS(const_cast<int*>(&back), b, final);
                return;
              }
              n = toPointer(b)->header.next;
              while(!isNull(n) && !isDeleted(n))
              {
                pdata->header.next = n;
                __threadfence();
                if(atomicCAS(const_cast<int*>(&toPointer(b)->header.next), n, offset) == n)
                  return;
                n = toPointer(b)->header.next;
              }
            }
            else
            {
              int nn = n;
              while(!isNull(nn) && equal(b,back))
              {
                n = nn;
                nn = toPointer(nn)->header.next;
              }
              atomicCAS(const_cast<int*>(&back), b, cleanOffset(n));
            }
          }
        }
      }
    }

    __inline__ __device__ volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>* prepareDequeue(int& f, int& n, int &iter, int &hops)
    {
      n = -1;

      while(true)
      {
        f = front;
        int b = back;
        n = toPointer(f)->header.next;
        if(equal(f,front))
        {
          if(equal(f,b))
          {
            if(isNull(n))
              return nullptr;
            int nn = n;
            while(!isNull(nn) && equal(back, b))
            {
              n = nn;
              nn = toPointer(nn)->header.next;
            }
            atomicCAS(const_cast<int*>(&back), b, cleanOffset(n));
          }
          else
          {
            iter = front;
            hops = 0;
            while(!isNull(n)  && isDeleted(n) && !equal(iter, back) && equal(f, front))
            {
              iter = n;
              n = toPointer(iter)->header.next;
              ++hops;
            }
            if(!equal(front,f))
              continue;
            else if(equal(iter,back))
              free_chain(front, iter);
            else
            {
              return toPointer(n);
            }
          }
        }
      }
    }

    __inline__ __device__ bool finishDequeue(int f, int n, int iter, int hops)
    {
      if(atomicCAS(const_cast<int*>(&toPointer(iter)->header.next), n, n |Deleted)==n)
      {
        //printf("%d freeing %d \n",blockIdx.x, f);
        if(hops >= MaxHops)
          free_chain(f, n);
        return true;
      }
      //printf("%d cas failed on %d \n",blockIdx.x, f);
      return false;
    }

    __inline__ __device__ void free_chain(int f, int nf)
    {
      if(atomicCAS(const_cast<int*>(&front), f, cleanOffset(nf)) == f)
      {
        while(!equal(f,nf))
        {
          int n = toPointer(f)->header.next;
          freeOffset(cleanOffset(f),toPointer(f)->myRealSize());
          f = n;
        }
      }
    }

  public:
    static std::string name()
    {
      return "BasketQueue";
    }

  };





  
  /***********************************************/
  // IMPLEMENTATION
  /***********************************************/
    

  template<uint TElementSize, uint TQueueSize, class TAdditionalData, bool TWarpOptimization,  template<uint > class MemAlloc = MemoryAllocFastest>
  class QueueLinkedBasketImp : public QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>, public ::BasicQueue<TAdditionalData>
  {

  public:
    static std::string name()
    {
      return QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>::name();
    }

    __inline__ __device__ void init() 
    {
      QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>::init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additionalData) 
    {
      return enqueue<Data>(data, additionalData);
    }

    template<class Data>
    __device__ bool enqueue(Data data, TAdditionalData additionalData) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata = 
        QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template prepareEnqueue<1, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeData(data, additionalData);
      
      QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template finishEnqueue<1,Data>(offset, pdata);
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data, TAdditionalData additionalData) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), TAdditionalData>* pdata = 
        QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template prepareEnqueue<Threads, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeDataParallel<Threads>(data, additionalData);
      
      QueueLinkedBasketBase<TElementSize, TQueueSize, TAdditionalData, TWarpOptimization, MemAlloc>:: template finishEnqueue<Threads,Data>(offset, pdata);
      return true;
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
          if(!startDequeue())
          {
            take = i;
            i = num;
            break;
          }

          int f, n, iter, hops;
          while(true)
          {
            volatile QueueLinkedStoreElement<TElementSize, TAdditionalData>* readdata = prepareDequeue(f, n, iter, hops);
            if(n == -1)
            {
              failDequeue();
              take = i;
              i = num;
              break;
            }
            int s = readdata->header.size;
            readdata->readData(cdata, additionalData + i);
            __threadfence();
            if(finishDequeue(f, n, iter, hops))
            {
              cdata += s/sizeof(uint);
              break;
            }
          }
        }
      }
      __syncthreads();
      return take;
    }
  };
  

  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization,  template<uint > class MemAlloc>
  class QueueLinkedBasketImp<TElementSize,TQueueSize,void,TWarpOptimization,MemAlloc> : public QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>, public ::BasicQueue<void>
  {

  public:
    static std::string name()
    {
      return QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>::name(); 
    }

    __inline__ __device__ void init() 
    {
      QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>::init();
    }

    template<class Data>
    __inline__ __device__ bool enqueueInitial(Data data) 
    {
      return enqueue<Data>(data);
    }

    template<class Data>
    __device__ bool enqueue(Data data) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), void>* pdata = 
        QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template prepareEnqueue<1, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeData(data);
      __threadfence();
      
      QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template finishEnqueue<1,Data>(offset, pdata);
      //if(threadIdx.x%32 == 0)
      //  printf("%d %d enqueue succeeded\n",blockIdx.x, threadIdx.x);
      return true;
    }
    
    template<int Threads, class Data>
    __device__ bool enqueue(Data* data) 
    {        
      int offset;
      volatile QueueLinkedStoreElement<sizeof(Data), void>* pdata = 
        QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template prepareEnqueue<Threads, Data>(offset);
      if(pdata == nullptr)
        return false;
      
      pdata->writeDataParallel<Threads>(data);
      __threadfence();
      
      QueueLinkedBasketBase<TElementSize, TQueueSize, void, TWarpOptimization, MemAlloc>:: template finishEnqueue<Threads,Data>(offset, pdata);
      return true;
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
          //printf("%d start dequeue %d\n",blockIdx.x,i);
          if(!startDequeue())
          {
            take = i;
            i = num;
            //printf("%d dequeue out 0\n",blockIdx.x);
            break;
          }
          int f, n, iter, hops;
          while(true)
          {
            volatile QueueLinkedStoreElement<TElementSize, void>* readdata = prepareDequeue(f, n, iter, hops);
            if(n == -1)
            {
              //printf("%d dequeue out 1\n",blockIdx.x);
              failDequeue();
              take = i;
              i = num;
              break;
            }
            //printf("%d read data %llx\n",blockIdx.x,readdata);
            readdata->readData(cdata);
            int s = readdata->header.size;
            __threadfence();
            if(finishDequeue(f, n, iter, hops))
            {
              //printf("(%d read data %llx (%d) succeeded\n",blockIdx.x,readdata,i);
              cdata += s/sizeof(uint);
              break;
            }
            //else
             // printf("%d could not read: %llx (%d)\n",blockIdx.x, readdata,i);
          }
        }
      }
      __syncthreads();
      //if(threadIdx.x == 1)
      //  printf("%d dequeue succeeded with: %d\n",blockIdx.x, take);
      return take;
    }
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedBasketFastAlloc : public QueueLinkedBasketImp<TElementSize, TQueueSize, TAdditionalData, false, MemoryAllocFastest>
  {
  public:
    static std::string name()
    {
      return "BasketQueueFast";
    }
  };

    template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedBasketOptFastAlloc : public QueueLinkedBasketImp<TElementSize, TQueueSize, TAdditionalData, true, MemoryAllocFastest>
  {
  public:
    static std::string name()
    {
      return "BasketQueueFastWarpoptimized";
    }
  };

  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedBasket : public QueueLinkedBasketImp<TElementSize, TQueueSize, TAdditionalData, false, MemoryAlloc>
  {
  public:
    static std::string name()
    {
      return "BasketQueue";
    }
  };
  template<uint TElementSize, uint TQueueSize, class TAdditionalData>
  class QueueLinkedBasketOpt : public QueueLinkedBasketImp<TElementSize, TQueueSize, TAdditionalData, true, MemoryAlloc>
  {
  public:
    static std::string name()
    {
      return "BasketQueueWarpoptimized";
    }
  };

  