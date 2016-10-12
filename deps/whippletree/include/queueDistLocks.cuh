#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "queueHelpers.cuh"
#include "segmentedStorage.cuh"
#include "bitonicSort.cuh"

  template<uint TQueueSize, bool TWarpOptimization = true, bool TAssertOnOverflow = true, bool TWithFence = false>
  class QueueDistLocksStub
  {
  protected:
    static const uint QueueSize = TQueueSize;

    uint front, back;
    volatile int count;
    volatile uint locks[QueueSize];

    int dummy0[4]; 

    volatile uint sortingFence;
    volatile uint hitSortingFence;
    uint sortingMinBorder;
    uint lastSortEnd;

    int dummy1[4]; 


    static std::string name()
    {
      return TWarpOptimization?"DistLocksWarpoptimized":"DistLocks";
    }
    
    __inline__ __device__ void init() 
    {
      uint lid = threadIdx.x + blockIdx.x*blockDim.x;
      if(lid == 0)
      {
        front = 0, back = 0, count = 0;
        if(TWithFence)
          sortingFence = QueueSize, hitSortingFence = 0, sortingMinBorder = 32, lastSortEnd = 0;
      }
      for(uint i = lid; i < QueueSize; i+=blockDim.x*gridDim.x)
        locks[i] = 0;
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
          int c = atomicAdd(const_cast<int*>(&count), ourcount);
          if(c + static_cast<int>(ourcount) < static_cast<int>(QueueSize))
            wpos = atomicAdd(&back, ourcount);
          else
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements %d\n", c);
              //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
              Softshell::trap();
            }
            atomicSub(const_cast<int*>(&count), ourcount);
          }
        }

        //get source
        int src = __ffs(mask)-1;
        //wpos = __shfl(wpos, src);
        wpos = warpBroadcast<32>(static_cast<unsigned int>(wpos), src);

        if(wpos == -1)
          return make_int2(-1,0);
        uint pos = (wpos + mypos/TthreadsPerElement)%QueueSize;
        while(locks[pos] != 0)
          __threadfence();
        return make_int2(pos, ourcount);
      }
      else
      {
        if(TthreadsPerElement == 1)
        {
          int c = atomicAdd(const_cast<int*>(&count), 1);
          if(c + 1 < static_cast<int>(QueueSize) )
          {
            uint pos = atomicAdd(&back, 1) % QueueSize;;
            while(locks[pos] != 0)
              __threadfence();
            return make_int2(pos, 1);
          }
          else
          {
            if(TAssertOnOverflow)
            {
              printf("ERROR queue out of elements\n");
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
          if(Softshell::laneid() % TthreadsPerElement == 0)
          {
            int c = atomicAdd(const_cast<int*>(&count), 1);
            if(c + 1 < static_cast<int>(QueueSize) )
            {
              pos = atomicAdd(&back, 1) % QueueSize;
              while(locks[pos] != 0)
                __threadfence();
            }
            else
            {
              if(TAssertOnOverflow)
              {
                printf("ERROR queue out of elements\n");
                //printf("ERROR queue out of elements %d+%d .. %d >%d\n", wpos, ourcount, wpos + ourcount - *static_cast<volatile uint*>(&front), QueueSize);
                Softshell::trap();
              }
              atomicSub(const_cast<int*>(&count), 1);
              pos = -1;
            }
          }

          //pos = __shfl(pos, 0, TthreadsPerElement);
          pos = warpBroadcast<TthreadsPerElement>(static_cast<unsigned int>(pos), 0);
          if(pos != -1)
            return make_int2(pos, 1);
          else
            return make_int2(pos, 0);
        }
      }
    }

    template<int TthreadsPerElement>
    __inline__ __device__  void enqueueEnd(int2 pos_ourcount)
    {
      if(TthreadsPerElement == 1)
        locks[pos_ourcount.x] = 1;
      else if(Softshell::laneid() % TthreadsPerElement == 0)
        locks[pos_ourcount.x] = 1;
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
        //else
        //  offset_take.x = 0;
      }
      __syncthreads();
      if(threadIdx.x < offset_take.y)
      {
        uint p = (offset_take.x + threadIdx.x)%QueueSize;
        while(locks[p] != 1)
          __threadfence();

        if(TWithFence)
        {
          uint currentfence;
          while((currentfence = sortingFence) != QueueSize)
          {
            if(currentfence > p) break;
            if(currentfence < back &&  back < p ) break;
            //ouch, we are blocked due to sorting!
            hitSortingFence = true;
            __threadfence();
          }
        }
      }

      return offset_take;
    }

    __inline__ __device__ void dequeueEnd(uint2 offset_take)
    {
      if(threadIdx.x < offset_take.y)
      {
        locks[(offset_take.x + threadIdx.x)%QueueSize] = 0;
        //__threadfence();
      }
    }

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      __shared__ int num;
      if(threadIdx.x == 0)
      {
        int c = atomicSub(const_cast<int*>(&count), maxnum);
        if(c < maxnum)
        {
          if(only_read_all)
          {
            atomicAdd(const_cast<int*>(&count), maxnum);
            num = 0;
          }
          else
          {
            atomicAdd(const_cast<int*>(&count), min(maxnum,maxnum - c));
            num = max(c, 0);
          }
        }
        else
          num = maxnum;
      }
      __syncthreads();
      return num;
    }
    __inline__ __device__ int startRead(int num)
    {
      __shared__ int offset;
      if(num <= 0)
        return 0;
      if(threadIdx.x == 0)
         offset = atomicAdd(&front, num);
      __syncthreads();
      if(threadIdx.x < num)
      {
        int pos = (offset + threadIdx.x)%QueueSize;
        while(locks[pos] != 1)
          __threadfence();
       
        if(TWithFence)
        {
          uint currentfence;
          while((currentfence = sortingFence) != QueueSize)
          {
            if(currentfence > pos) break;
            if(currentfence < back &&  back < pos ) break;
            //ouch, we are blocked due to sorting!
            hitSortingFence = true;
            __threadfence();
          }
        }
      }
      return offset;
    }
    __inline__ __device__ void finishRead(int id, int num)
    {
      if(threadIdx.x < num)
        locks[(id + threadIdx.x)%QueueSize] = 0;
    }

  public:
    __inline__ __device__ int size() const
    {
      return count;
    }
    __inline__ __device__ void clear()
    {
      if (blockIdx.x == 0 && threadIdx.x == 0)
        front = 0, back = 0, count = 0;
      for (uint i = blockIdx.x * blockDim.x + threadIdx.x; i < QueueSize; i += blockDim.x*gridDim.x)
        locks[i] = 0;
    }
  };



  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
  class QueueDistUnequalLocks : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, AllocStorage<TElementSize, TAdditionalData, TQueueSize> >
  {
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow>
  class QueueDistUnequalLocks<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow> : public QueueBuilder<TElementSize, TQueueSize, void, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, AllocStorage<TElementSize, void, TQueueSize> >
  {
  };


  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
  class QueueDistLocks : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> >
  {
  public:

    template<class Data>
    __inline__ __device__ Data* reserveSpot(TAdditionalData additionalData)
    {
      uint pos = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>:: template enqueuePrep<1>(make_int2(0,0)).x;
      QueueStorage<TElementSize, TAdditionalData, TQueueSize>::writeAdditionalData(additionalData, pos);
      return reinterpret_cast<Data*>(QueueStorage<TElementSize, TAdditionalData, TQueueSize>::readDataPointer(pos));
    }

    template<int threads, class Data>
    __inline__ __device__ Data* reserveSpot(TAdditionalData additionalData)
    {
      uint pos = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>:: template enqueuePrep<threads>(make_int2(0,0)).x;
      QueueStorage<TElementSize, TAdditionalData, TQueueSize>:: template writeAdditionalDataParallel<threads>(additionalData, pos);
      return reinterpret_cast<Data*>(QueueStorage<TElementSize, TAdditionalData, TQueueSize>::readDataPointer(pos));
    }

    template<class Data>
    __inline__ __device__ void completeSpot(Data* spot)
    {
      uint pos = QueueStorage<TElementSize, TAdditionalData, TQueueSize>::getPosFromPointer(spot);
      __threadfence();
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::  template enqueueEnd<1>(make_int2(pos,1));
    }

    template<int threads, class Data>
    __inline__ __device__ void completeSpot(Data* spot)
    {
      uint pos = QueueStorage<TElementSize, TAdditionalData, TQueueSize>::getPosFromPointer(spot);
      __threadfence();
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::  template enqueueEnd<threads>(make_int2(pos,1));
    }

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::reserveRead(maxnum, only_read_all);
    }
    __inline__ __device__ int startRead(void*& data, TAdditionalData* addtionalData, int pos, int num)
    {
      int offset = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::startRead(num);
      if(pos < num)
        data = QueueStorage<TElementSize, TAdditionalData, TQueueSize>::readDataPointers(addtionalData + pos, offset + pos);
      return offset;
    }
    __inline__ __device__ void finishRead(int id, int num)
    {
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::finishRead(id, num);
      QueueStorage<TElementSize, TAdditionalData, TQueueSize>::storageFinishRead(make_uint2(id,num) );
    }
  };
  
  template<uint TElementSize, uint TQueueSize, bool TWarpOptimization, bool TAssertOnOverflow>
  class QueueDistLocks<TElementSize, TQueueSize, void, TWarpOptimization, TAssertOnOverflow> : public QueueBuilder<TElementSize, TQueueSize, void, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, QueueStorage<TElementSize, void, TQueueSize> >
  {
     public:

    template<class Data>
    __inline__ __device__ Data* reserveSpot()
    {
      uint pos = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::template enqueuePrep<1>(make_int2(0,0)).x;
      return reinterpret_cast<Data*>(QueueStorage<TElementSize, void, TQueueSize>::  readDataPointer(pos));
    }

    template<int threads, class Data>
    __inline__ __device__ Data* reserveSpot()
    {
      uint pos = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>:: template enqueuePrep<threads>(make_int2(0,0)).x;
      return reinterpret_cast<Data*>(QueueStorage<TElementSize, void, TQueueSize>::readDataPointer(pos));
    }

    template<class Data>
    __inline__ __device__ void completeSpot(Data* spot)
    {
      uint pos = QueueStorage<TElementSize, void, TQueueSize>::getPosFromPointer(spot);
      __threadfence();
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>:: template enqueueEnd<1>(make_int2(pos,1));
    }

    template<int threads, class Data>
    __inline__ __device__ void completeSpot(Data* spot)
    {
      uint pos = QueueStorage<TElementSize, void, TQueueSize>::getPosFromPointer(spot);
      __threadfence();
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>:: template enqueueEnd<threads>(make_int2(pos,1));
    }

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::reserveRead(maxnum, only_read_all);
    }
    __inline__ __device__ int startRead(void*& data, int pos, int num)
    {
      int offset = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::startRead(num);
      if(pos < num)
        data = QueueStorage<TElementSize, void, TQueueSize>::readDataPointers(offset + pos);
      return offset;
    }
    __inline__ __device__ void finishRead(int id, int num)
    {
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::finishRead(id, num);
      QueueStorage<TElementSize, void, TQueueSize>::storageFinishRead(make_uint2(id,num) );
    }
  };


  
  template<uint TElementSize, uint TQueueSize, class TAdditionalData, class ExternalStorage, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
  class QueueDistLocksExternal : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, SegmentedStorage::SegmentedQueueStorage<TElementSize, TAdditionalData, TQueueSize, ExternalStorage> >
  {
  public:

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::reserveRead(maxnum, only_read_all);
    }
    __inline__ __device__ int startRead(void*& data, TAdditionalData* addtionalData, int pos, int num)
    {
      int offset = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::startRead(num);
      if(pos < num)
        data = SegmentedStorage::SegmentedQueueStorage<TElementSize, TAdditionalData, TQueueSize, ExternalStorage>::readDataPointers(addtionalData + pos, offset + pos);
      return offset;
    }
    __inline__ __device__ void finishRead(int id, int num)
    {
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::finishRead(id, num);
      SegmentedStorage::SegmentedQueueStorage<TElementSize, TAdditionalData, TQueueSize, ExternalStorage>::storageFinishRead(make_uint2(id,num) );
    }
  };

  template<uint TElementSize, uint TQueueSize, class ExternalStorage, bool TWarpOptimization, bool TAssertOnOverflow>
  class QueueDistLocksExternal<TElementSize, TQueueSize, void, ExternalStorage, TWarpOptimization, TAssertOnOverflow> : public QueueBuilder<TElementSize, TQueueSize, void, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>, SegmentedStorage::SegmentedQueueStorage<TElementSize, void, TQueueSize, ExternalStorage> >
  {
     public:

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::reserveRead(maxnum, only_read_all);
    }
    __inline__ __device__ int startRead(void*& data, int pos, int num)
    {
      int offset = QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::startRead(num);
      if(pos < num)
        data = SegmentedStorage::SegmentedQueueStorage<TElementSize, void, TQueueSize, ExternalStorage>::readDataPointers(offset + pos);
      return offset;
    }
    __inline__ __device__ void finishRead(int id, int num)
    {
      QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow>::finishRead(id, num);
      SegmentedStorage::SegmentedQueueStorage<TElementSize, void, TQueueSize, ExternalStorage>::storageFinishRead(make_uint2(id,num) );
    }
  };



  template<uint TElementSize, uint TQueueSize, class TAdditionalData = void, bool TWarpOptimization = true, bool TAssertOnOverflow = true>
  class QueueDistLocksSortable : public QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, true>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> >
  {
    typedef QueueBuilder<TElementSize, TQueueSize, TAdditionalData, QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, true>, QueueStorage<TElementSize, TAdditionalData, TQueueSize> > Base;
    typedef QueueDistLocksStub<TQueueSize, TWarpOptimization, TAssertOnOverflow, true> Stub;
    typedef QueueStorage<TElementSize, TAdditionalData, TQueueSize> Storage;


    typedef typename StorageElementTyping<TElementSize>::Type QueueData_t;
  public:

    __inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
    {
      return Stub::reserveRead(maxnum, only_read_all);
    }
    __inline__ __device__ int startRead(void*& data, TAdditionalData* addtionalData, int pos, int num)
    {
      int offset = Stub::startRead(num);
      if(pos < num)
        data = Stub::readDataPointers(addtionalData + pos, offset + pos);
      return offset;
    }
    __inline__ __device__ void finishRead(int id, int num)
    {
      Stub::finishRead(id, num);
      Storage::storageFinishRead(make_uint2(id,num) );
    }
      


    template<class SortInfo>
    __inline__ __device__  bool sort( unsigned int threads)
    {
      __shared__ int sortStart;
      extern __shared__ uint s_data[];

      uint num = 2*threads;
      uint linId = threadIdx.x;

      int cFront = 0;
      if(linId == 0)
      {
        cFront = *((volatile uint*)(&this->front));
        int cBack = *((volatile uint*)(&this->back))%this->QueueSize;

        int thisSortEnd = this->lastSortEnd;
        //construct not ringbuffered
        if(cFront > cBack)
        {
          cBack += this->QueueSize;
          if(cBack > thisSortEnd)
            thisSortEnd += this->QueueSize;
        }

        //compute next sorting position
        if(thisSortEnd == this->QueueSize || thisSortEnd < cFront)
          thisSortEnd = cBack - (int)num;
        else
          thisSortEnd = this->lastSortEnd - (num/2);

        //is there enough border?
        int maxfill = thisSortEnd - (int)this->sortingMinBorder;
        if(maxfill < cFront || this->count < (int)(256 + this->sortingMinBorder + num))
        {
          this->lastSortEnd = this->QueueSize;
          sortStart = -1;
        }
        else
        {
          sortStart = thisSortEnd;
        }

        ////debug
        //if(sortStart < 0)
        //  printf("not going to sort %d (%d->%d = %d)\n", sortStart, cFront, cBack, *(volatile int*)&count);
        //else
        //{
        //  printf("going to try sort @%d (%d->%d = %d)!\n", sortStart, cFront, cBack, *(volatile int*)&count);
        //  lastSortEnd = sortStart;
        //}
      }


      Softshell::syncthreads(1, threads);
      if(sortStart < 0) return false;

      ////debug
      //clock_t startLoad = getTimeCycles();
      ////debug

      //load in data
      for(uint i = linId; i < num; i += threads)
      {
        uint elementId = (sortStart + i) % this->QueueSize;
        while(this->locks[elementId] == 0)
          __threadfence();
       
        int addInfo;
        void * data = Storage::readDataPointers(&addInfo, elementId);

        s_data[i] = elementId;
        s_data[i + 2*threads] = SortInfo::eval(addInfo, data);
      }

      __threadfence();
      Softshell::syncthreads(1, threads);

      ////debug
      //clock_t endLoad = getTimeCycles();
      ////debug

      //check if still ok and enable fence
      if(linId == 0)
      {
        this->hitSortingFence = false;
        this->sortingFence = sortStart % this->QueueSize;
        __threadfence();
        int nFront = *((volatile uint*)(&this->front));
        if(nFront < cFront) nFront += this->QueueSize;

        int maxfill = sortStart - (int)this->sortingMinBorder/2;
        if(maxfill < nFront)
        {
          //outch not enough space left
          this->sortingFence = this->QueueSize;
          this->lastSortEnd = this->QueueSize;
          sortStart = -1;
        }
        else
          this->lastSortEnd = sortStart;

        ////debug
        //if(sortStart < 0)
        //  printf("disabled fence %d (%d/%d->%d = %d)\n", sortStart, cFront, nFront, back, *(volatile int*)&count);
        //else
        //{
        //  printf("fence is up @%d (%d/%d->%d = %d)!\n", sortStart, cFront, nFront, back, *(volatile int*)&count);
        //  lastSortEnd = sortStart;
        //}
      }

      ////deb
      //sortingFence = QueueSize;
      //return;
      ////deb

      Softshell::syncthreads(1, threads);
      if(sortStart < 0) return false;

      ////debug
      //clock_t startSort = getTimeCycles();
      ////debug

      //sort
      if(linId < num/2)
        Softshell::Sort::bitonic<uint, uint, false>(s_data+2*threads, s_data, linId, num);
      Softshell::syncthreads(2, threads);


      ////debug
      //clock_t endSort = getTimeCycles();
      ////debug


      //copy in
      TAdditionalData addOne, addTwo;
      QueueData_t dataOne, dataTwo;

      Storage::readData((void*)(&dataOne), &addOne, s_data[linId]);
      Storage::readData((void*)(&dataTwo), &addTwo, s_data[linId + threads]);
      Softshell::syncthreads(1, threads);
      Storage::template writeData<QueueData_t>(dataOne, addOne, make_uint2((sortStart + linId) % this->QueueSize, 0));
      Storage::template writeData<QueueData_t>(dataTwo, addTwo, make_uint2((sortStart + linId + threads) % this->QueueSize, 0));


      //write out
      //if(threadIdx.x == 0)
      //{
      //  for(int i = 0; i < num; ++i)
      //    printf("%d; ", ids[i]);
      //  printf("\n");
      //  for(int i = 0; i < num; ++i)
      //    printf("%d; ", priorities[i]);
      //  printf("\n");
      //}


      __threadfence();
      Softshell::syncthreads(1, threads);

      ////debug
      //clock_t endWrite = getTimeCycles();
      ////debug


      //unset fence
      if(linId == 0)
      {
        this->sortingFence = this->QueueSize;
        ////debug
        //printf("sorting done %d->%d queue: %d->%d, sorting: %d->%d (l: %d, s: %d, w: %d) %d\n", startLoad, endWrite, front, back, sortStart, sortStart+num, endLoad-startLoad, endSort-startSort, endWrite-endSort,hitSortingFence);
        //printf("sorting done queue: %d->%d, sorting: %d->%d\n",front, back, sortStart, sortStart+num);

        if(this->hitSortingFence)
        {
          //we need to increase the margin
          this->sortingMinBorder += 64;
          this->hitSortingFence = false;
        }
      }
      return true;
    }
  };

template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueDistLocks_t : public QueueDistLocks<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueDistLocksOpt_t : public QueueDistLocks<TElementSize, TQueueSize, TAdditionalData, true,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueDistLocksNoOverflow_t : public QueueDistLocks<TElementSize, TQueueSize, TAdditionalData, false,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueDistLocksNoOverflowOpt_t : public QueueDistLocks<TElementSize, TQueueSize, TAdditionalData, true,false> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueDistLocksSortable_t : public QueueDistLocksSortable<TElementSize, TQueueSize, TAdditionalData, false,true> { };
template<uint TElementSize, uint TQueueSize, class TAdditionalData> class QueueDistLocksSortableOpt_t : public QueueDistLocksSortable<TElementSize, TQueueSize, TAdditionalData, true,true> { };
