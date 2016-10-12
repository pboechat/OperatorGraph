#pragma once

#include "common.cuh"

//#define DEBUG_STORAGE

namespace SegmentedStorage
{

  void (*pReinitStorage)() = 0;
  __device__ void* storage;
  template<int TStorageSize, int TBlockSize>
  class Storage
  {
  public:
    static const int StorageSize = TStorageSize;
    static const int BlockSize = TBlockSize/16*16;
    struct Block
    {
      unsigned int data[BlockSize/sizeof(int)];
    };

  private:
    static const int NumBlocks = (StorageSize - 16*sizeof(uint)) / (BlockSize + sizeof(int));
    
    Block blocks[NumBlocks];
    
    int count;
    unsigned int front, back;
    volatile int available[NumBlocks];
  public:
    
    __inline__ __device__ void init()
    {
      count = NumBlocks;
      back = front = 0;
      for(int id = threadIdx.x + blockIdx.x*blockDim.x; id < NumBlocks; id += gridDim.x*blockDim.x)
        available[id] = id;
    }

    __inline__ __device__ int request()
    {
      int c = atomicSub(&count, 1);
      if(c <= 0)
      {
        atomicAdd(&count, 1);
        return -1;
      }
      int p = atomicInc(&front, NumBlocks-1);
      int id;
      while( (id = available[p]) == -1) 
        __threadfence();
      available[p] = -1;
      return id;
    }

    __inline__ __device__ void free(int id)
    {
      int p = atomicInc(&back, NumBlocks-1);
      while(available[p] != -1)
        __threadfence();
      available[p] = id;
      __threadfence();
      atomicAdd(&count, 1);
    }

    __inline__ __device__ void free(Block* b)
    {
      free(blockToIndex(b));
    }

    __inline__ __device__ Block* indexToBlock(int index)
    {
      //printf("convert: (blocks: %ull) %d/%d->%ull\n",blocks,index,NumBlocks,blocks + index);
      return blocks + index;
    }

    __inline__ __device__ int blockToIndex(Block* b)
    {
      return b - blocks;
    }

    static __inline__ __device__ Storage* get()
    {
      return reinterpret_cast<Storage<StorageSize, BlockSize>*>(storage);
    }
  };

  void* StoragePointer = 0;

  template<int StorageSize, int BlockSize>
  __global__ void initStorage(void* data)
  {
    storage = data;
    Storage<StorageSize, BlockSize>* s = reinterpret_cast<Storage<StorageSize, BlockSize>*>(data);
    s->init();
  }

  template<class Storage>
  void reinitStorage()
  {
    if(StoragePointer == 0)
      CUDA_CHECKED_CALL(cudaMalloc(&StoragePointer, Storage::StorageSize));
    initStorage<Storage::StorageSize, Storage::BlockSize><<<512,512>>>(StoragePointer);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());
  }


  template<class Storage>
  void createStorage()
  {
    CUDA_CHECKED_CALL(cudaMalloc(&StoragePointer, Storage::StorageSize));
    initStorage<Storage::StorageSize, Storage::BlockSize><<<512,512>>>(StoragePointer);
    CUDA_CHECKED_CALL(cudaDeviceSynchronize());
    pReinitStorage = &reinitStorage<Storage>;
  }

  void destroyStorage()
  {
    if(StoragePointer != 0)
      CUDA_CHECKED_CALL(cudaFree(&StoragePointer));
    StoragePointer = 0;
    pReinitStorage = 0;
  }

  void checkReinitStorage()
  {
    if(pReinitStorage != 0)
      pReinitStorage();
  }
 

  template<uint TQueueSize, uint ElementsPerBlock, class SharedStorage>
  class SegmentedQueueStorageBase
  {
  protected:

    static const int MaxBlocks = (TQueueSize+ElementsPerBlock-1)/ElementsPerBlock;

    template<class QueueData_T, class QueueAddtionalData_T>
    struct MyBlock
    {
      volatile QueueData_T storage[ElementsPerBlock];
      volatile QueueAddtionalData_T additionalStorage[ElementsPerBlock];
      int available;
      __inline__ __device__ void init()
      {
        *(volatile int*)(&available) = ElementsPerBlock;
      }
      __inline__ __device__ void use(int num = 1)
      {
      }
      __inline__ __device__ bool doneuse(int num = 1)
      {
        return atomicSub(&available, num) <= num;
      }
    };

    template<class QueueData_T>
    struct MyBlock<QueueData_T, void>
    {
      volatile QueueData_T storage[ElementsPerBlock];
      int available;
      __inline__ __device__ void init()
      {
        *(volatile int*)(&available) = ElementsPerBlock;
      }
      __inline__ __device__ void use(int num = 1)
      {
      }
      __inline__ __device__ bool doneuse(int num = 1)
      {
        return  atomicSub(&available, num) <= num;
      }
    };

    volatile int useBlocks[MaxBlocks];

    template<class TMyBlock, int Smaller>
     __inline__ __device__  TMyBlock* acquireBlock(int pos)
    {
      int block = pos / ElementsPerBlock;
      int blockoffset = useBlocks[block];
      if(blockoffset == -1)
      {
        int localpos = pos - ElementsPerBlock*block;
        if(Smaller == 0 || localpos < Smaller)
        {
          blockoffset = SharedStorage::get()->request();
#ifdef DEBUG_STORAGE
          printf("%d %d requests new block for %d(%d), got %d \n",blockIdx.x, threadIdx.x, block, pos, blockoffset);
#endif
          if(blockoffset == -1)
          {
            __threadfence();
            blockoffset = useBlocks[block];
            if(blockoffset == -1)
            {
#ifdef DEBUG_STORAGE
              printf("Error: SharedStorage out of elements\n");
#endif
              Softshell::trap();
            }
          }
          else
          {
            reinterpret_cast<TMyBlock*>(SharedStorage::get()->indexToBlock(blockoffset))->init();
            int oldid = atomicCAS((int*)(useBlocks + block), -1, blockoffset);
            if(oldid != -1)
            {
#ifdef DEBUG_STORAGE
              printf("%d %d putting block back: %d (requested by %d(%d))\n",blockIdx.x, threadIdx.x, blockoffset, block, pos);
#endif
              SharedStorage::get()->free(blockoffset);
              blockoffset = oldid;
            }
          }
        }
        else
        {
          while( (blockoffset = useBlocks[block]) == -1)
            __threadfence();
        }
      }
      return reinterpret_cast<TMyBlock*>(SharedStorage::get()->indexToBlock(blockoffset));
    }

    template<class TMyBlock>
    __inline__ __device__  TMyBlock* getBlock(int pos)
    {
      int block = pos / ElementsPerBlock;
      int blockoffset = useBlocks[block];
      return reinterpret_cast<TMyBlock*>(SharedStorage::get()->indexToBlock(blockoffset));
    }


    template<class TMyBlock>
    __inline__ __device__ void storageFinishRead(uint2 pos)
    {
      int mypos = (pos.x + threadIdx.x) % TQueueSize;;
      int prevblock = (mypos-1)/ElementsPerBlock;
      int myblock = mypos/ElementsPerBlock;
      if(threadIdx.x < pos.y && (threadIdx.x == 0 || (myblock != prevblock)) )
      {
        int donelements = min((int)((myblock + 1)*ElementsPerBlock - mypos),(int)(pos.y-threadIdx.x));
        int bid = useBlocks[myblock];
        TMyBlock* b = reinterpret_cast<TMyBlock*>(SharedStorage::get()->indexToBlock(bid));
        if(b->doneuse(donelements))
        {
          useBlocks[myblock] = -1;
#ifdef DEBUG_STORAGE
          printf("%d %d freeing: %d\n",blockIdx.x, threadIdx.x, bid);
#endif
          SharedStorage::get()->free(bid);
        }
      }
    }

  public:

   

    static std::string name()
    {
      return "SharedStorage";
    }
  
    __inline__ __device__ void init()
    {
      int id = blockIdx.x*blockDim.x + threadIdx.x;
      for(int i = id; i < MaxBlocks; i+= blockDim.x*gridDim.x)
        useBlocks[i] = -1;
#ifdef DEBUG_STORAGE
  if(id == 0)
        printf("maxblocks: %d\n",MaxBlocks);
#endif
    }

  };


  template<uint TElementSize, class TAdditionalData, uint TQueueSize, class SharedStorage>
  class SegmentedQueueStorage : public SegmentedQueueStorageBase<TQueueSize, (SharedStorage::BlockSize - sizeof(uint)) / (sizeof(TAdditionalData) + TElementSize), SharedStorage>
  {
  protected:
    typedef typename StorageElementTyping<TElementSize>::Type QueueData_T;
    typedef typename StorageElementTyping<sizeof(TAdditionalData)>::Type QueueAddtionalData_T;

    static const int ElementsPerBlock = (SharedStorage::BlockSize - sizeof(uint)) / (sizeof(TAdditionalData) + TElementSize);
    typedef SegmentedQueueStorageBase<TQueueSize, ElementsPerBlock, SharedStorage> Base;
    typedef typename Base::MyBlock<QueueData_T, QueueAddtionalData_T> MyBlock;

    

  public:

    template<class T>
    __inline__ __device__ uint prepareData(T data, TAdditionalData additionalData)
    {
      return 0;
    }

    template<int TThreadsPerElenent, class T>
    __inline__ __device__ uint prepareDataParallel(T* data, TAdditionalData additionalData)
    {
      return 0;
    }

    template<class T>
    __inline__ __device__ void writeData(T data, TAdditionalData additionalData, uint2 pos)
    {
      pos.x = pos.x%TQueueSize;
      int localpos = pos.x % ElementsPerBlock;
      MyBlock* b = Base:: template acquireBlock<MyBlock, 32>(pos.x);
      b->use();

      b->storage[localpos] = *reinterpret_cast<QueueData_T*>(&data);
      b->additionalStorage[localpos] = *reinterpret_cast<QueueAddtionalData_T*>(&additionalData);
    }

    template<int TThreadsPerElenent, class T>
    __inline__ __device__ void writeDataParallel(T* data, TAdditionalData additionalData, uint2 pos)
    {
      pos.x = pos.x%TQueueSize;
      int localpos = pos.x % ElementsPerBlock;
      MyBlock* b = Base:: template acquireBlock<MyBlock, 32>(pos.x);
      if(threadIdx.x % TThreadsPerElenent == 0)
        b->use();

      multiWrite<TThreadsPerElenent, T>(reinterpret_cast<volatile T*>(b->storage + localpos), data);
      multiWrite<TThreadsPerElenent, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(b->additionalStorage + localpos), &additionalData);

    }

    __inline__ __device__ void readData(void* data, TAdditionalData* additionalData, uint pos)
    {
      pos = pos%TQueueSize;
      int localpos = pos % ElementsPerBlock;
      MyBlock* b = Base:: template getBlock<MyBlock>(pos);
      *reinterpret_cast<QueueData_T*>(data) = b->storage[localpos];
      *reinterpret_cast<QueueAddtionalData_T*>(additionalData) = b->additionalStorage[localpos];
    }

    __inline__ __device__ void* readDataPointers(TAdditionalData* additionalData, uint pos)
    {
      pos = pos%TQueueSize;
      int localpos = pos % ElementsPerBlock;
      MyBlock* b = Base:: template getBlock<MyBlock>(pos);
      *reinterpret_cast<QueueAddtionalData_T*>(additionalData) = b->additionalStorage[localpos];
      return (void*)(b->storage + localpos);
    }

    __inline__ __device__ void storageFinishRead(uint2 pos)
    {
      Base:: template storageFinishRead<MyBlock>(pos);
    }
  };

  template<uint TElementSize, uint TQueueSize, class SharedStorage>
  class SegmentedQueueStorage<TElementSize, void, TQueueSize, SharedStorage> : public SegmentedQueueStorageBase<TQueueSize, (SharedStorage::BlockSize - sizeof(uint)) / (TElementSize), SharedStorage>
  {
  protected:
    typedef typename StorageElementTyping<TElementSize>::Type QueueData_T;

    static const int ElementsPerBlock = (SharedStorage::BlockSize - sizeof(uint)) / (TElementSize);
    typedef SegmentedQueueStorageBase<TQueueSize, ElementsPerBlock, SharedStorage> Base;
    typedef typename Base::MyBlock<QueueData_T, void> MyBlock;

  public:

    template<class T>
    __inline__ __device__ uint prepareData(T data)
    {
      return 0;
    }

    template<int TThreadsPerElenent, class T>
    __inline__ __device__ uint prepareDataParallel(T* data)
    {
      return 0;
    }

    template<class T>
    __inline__ __device__ void writeData(T data, uint2 pos)
    {
      pos.x = pos.x%TQueueSize;
      int localpos = pos.x % ElementsPerBlock;
      MyBlock* b = Base:: template acquireBlock<MyBlock, 32>(pos.x);
      b->use();
	  #ifdef DEBUG_STORAGE
		printf("block pointer: %llx, data pointer: %llx, mydata pointer %llx\n",b,b->storage,&b->storage[localpos]);
#endif
      store(b->storage[localpos],*reinterpret_cast<QueueData_T*>(&data));
      //b->storage[localpos] = *reinterpret_cast<QueueData_T*>(&data);
      //printf("wrote for %d @ %d (%d queuesize, %d elementsperblock) on %ull\n", pos.x, localpos,TQueueSize,ElementsPerBlock,b);
    }

    template<int TThreadsPerElenent, class T>
    __inline__ __device__ void writeDataParallel(T* data, uint2 pos)
    {
     pos.x = pos.x%TQueueSize;
      int localpos = pos.x % ElementsPerBlock;
      MyBlock* b = Base:: template acquireBlock<MyBlock, 32>(pos.x);
      if(threadIdx.x % TThreadsPerElenent == 0)
        b->use();

      multiWrite<TThreadsPerElenent, T>(reinterpret_cast<volatile T*>(b->storage + localpos), data);
    }

    __inline__ __device__ void readData(void* data, uint pos)
    {
      pos = pos%TQueueSize;
      int localpos = pos % ElementsPerBlock;
      MyBlock* b = Base:: template getBlock<MyBlock>(pos);
      load(*reinterpret_cast<QueueData_T*>(data), b->storage[localpos]);
      //*reinterpret_cast<QueueData_T*>(data) = b->storage[localpos];
    }

    __inline__ __device__ void* readDataPointers(uint pos)
    {
      pos = pos%TQueueSize;
      int localpos = pos % ElementsPerBlock;
      MyBlock* b = Base:: template getBlock<MyBlock>(pos);
      return (void*)(b->storage + localpos);
    }

    __inline__ __device__ void storageFinishRead(uint2 pos)
    {
      Base:: template storageFinishRead<MyBlock>(pos);
    }
  };
}





