#pragma once

#include "random.cuh"

template<class TAdditionalData>
struct AdditionalDataInfo
{
  static const int size = sizeof(TAdditionalData);
};

template<>
struct AdditionalDataInfo<void>
{
  static const int size = 0;
};

template<int Mod, int MaxWarps>
__device__ __inline__ uint warpBroadcast(uint val, int who)
{
#if __CUDA_ARCH__ < 300
	__shared__ volatile uint comm[MaxWarps];
  for(int offset = 0; offset < 32; offset += Mod)
  {
    if(Softshell::laneid() - offset == who)
      comm[threadIdx.x/32] = val;
    if(Softshell::laneid() < offset + Mod)
      return comm[threadIdx.x/32];
  }
  return val;
#else
   return __shfl(val, who, Mod);
#endif
}
template<int Mod>
__device__ __inline__ uint warpBroadcast(uint val, int who)
{
  return warpBroadcast<Mod,32>(val, who);
}

template<int Mod, int MaxWarps>
__device__ __inline__ int warpBroadcast(int val, int who)
{
	return static_cast<int>(warpBroadcast<Mod, MaxWarps>(static_cast<uint>(val), who));
}
template<int Mod>
__device__ __inline__ int warpBroadcast(int val, int who)
{
	return warpBroadcast<Mod, 32>(val, who);
}

template<int Mod, int MaxWarps>
__device__ __inline__ unsigned long long int warpBroadcast(unsigned long long int val, int who)
{
	uint hi = warpBroadcast<Mod, MaxWarps>(static_cast<unsigned int>(__double2hiint(__longlong_as_double(val))), who);
	uint lo = warpBroadcast<Mod, MaxWarps>(static_cast<unsigned int>(__double2loint(__longlong_as_double(val))), who);
	return __double_as_longlong(__hiloint2double(hi, lo));
}
template<int Mod>
__device__ __inline__ unsigned long long int warpBroadcast(unsigned long long int val, int who)
{
	return warpBroadcast<Mod, 32>(val, who);
}


template<int Mod, int MaxWarps, class T>
__device__ __inline__ T* warpBroadcast(T* val, int who)
{
	return reinterpret_cast<T*>(warpBroadcast<Mod, MaxWarps>(reinterpret_cast<Softshell::PointerEquivalent>(val), who));
}
template<int Mod, class T>
__device__ __inline__ T* warpBroadcast(T* val, int who)
{
	return warpBroadcast<Mod, 32>(val, who);
}


template<int Mod, int MaxWarps>
__device__ __inline__ int warpShfl(int val, int who)
{
#if __CUDA_ARCH__ < 300
  __shared__ volatile int comm[MaxWarps];
  int runid = 0;
  int res = val;
  for(int offset = 0; offset < 32; offset += Mod)
  {
    for(int within = 0; within < Mod; ++within)
    {
      if(Softshell::laneid() == runid)
        comm[threadIdx.x/32] = val;
      if( Softshell::laneid() >= offset 
        && Softshell::laneid() < offset + Mod 
        && (runid % Mod) == ((who + 32) % Mod) )
        res = comm[threadIdx.x/32];
      ++runid;
    }
  }
  return res;
#else
   return __shfl(val, who, Mod);
#endif
}
template<int Mod>
__device__ __inline__ int warpShfl(int val, int who)
{
  return warpShfl<Mod,32>(val, who);
}


template<int Maxrand>
__device__ __inline__ void backoff(int num)
{

  volatile int local = threadIdx.x;
  for(int i = 0; i < (qrandom::rand() % Maxrand); ++i)
  {
    local += num*threadIdx.x/(i+1234);
    __threadfence();
  }
}


__inline__ __device__ uint4& load(uint4& dest, const volatile uint4& src)
{
	asm("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(dest.x), "=r"(dest.y), "=r"(dest.z), "=r"(dest.w) : "l"(&src));
	return dest;
}

__inline__ __device__ uint2& load(uint2& dest, const volatile uint2& src)
{
	asm("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(dest.x), "=r"(dest.y) : "l"(&src));
	return dest;
}

__inline__ __device__ uint& load(uint& dest, const volatile uint& src)
{
	dest = src;
	return dest;
}

__inline__ __device__ uint1& load(uint1& dest, const volatile uint1& src)
{
	dest.x = src.x;
	return dest;
}

__inline__ __device__ uchar3& load(uchar3& dest, const volatile uchar3& src)
{
	dest.x = src.x;
	dest.y = src.y;
	dest.z = src.z;
	return dest;
}

__inline__ __device__ uchar2& load(uchar2& dest, const volatile uchar2& src)
{
	dest.x = src.x;
	dest.y = src.y;
	return dest;
}

__inline__ __device__ uchar1& load(uchar1& dest, const volatile uchar1& src)
{
	dest.x = src.x;
	return dest;
}


__inline__ __device__ volatile uint4& store(volatile uint4& dest, const uint4& src)
{
	asm("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(&dest), "r"(src.x), "r"(src.y), "r"(src.z), "r"(src.w));
	return dest;
}

__inline__ __device__ volatile uint2& store(volatile uint2& dest, const uint2& src)
{
	asm("st.volatile.global.v2.u32 [%0], {%1, %2};" : : "l"(&dest), "r"(src.x), "r"(src.y));
	return dest;
}

__inline__ __device__ volatile uint& store(volatile uint& dest, const uint& src)
{
	dest = src;
	return dest;
}

__inline__ __device__ volatile uint1& store(volatile uint1& dest, const uint1& src)
{
	dest.x = src.x;
	return dest;
}

__inline__ __device__ volatile uchar3& store(volatile uchar3& dest, const uchar3& src)
{
	dest.x = src.x;
	dest.y = src.y;
	dest.z = src.z;
	return dest;
}

__inline__ __device__ volatile uchar2& store(volatile uchar2& dest, const uchar2& src)
{
	dest.x = src.x;
	dest.y = src.y;
	return dest;
}

__inline__ __device__ volatile uchar1& store(volatile uchar1& dest, const uchar1& src)
{
	dest.x = src.x;
	return dest;
}



__inline__ __device__ void clear(volatile uint4& dest)
{
	asm("st.volatile.global.v4.u32 [%0], {0, 0, 0, 0};" : : "l"(&dest));
}

__inline__ __device__ void clear(volatile uint2& dest)
{
	asm("st.volatile.global.v2.u32 [%0], {0, 0};" : : "l"(&dest));
}

__inline__ __device__ void clear(volatile uint& dest)
{
	dest = 0;
}

__inline__ __device__ void clear(volatile uint1& dest)
{
	dest.x = 0;
}

__inline__ __device__ void clear(volatile uchar3& dest)
{
	dest.x = 0;
	dest.y = 0;
	dest.z = 0;
}

__inline__ __device__ void clear(volatile uchar2& dest)
{
	dest.x = 0;
	dest.y = 0;
}

__inline__ __device__ void clear(volatile uchar1& dest)
{
	dest.x = 0;
}


template<uint TElementSize>
struct StorageElement16
{
	static const int num_storage_owords = (TElementSize + 15) / 16;

	uint4 storage[num_storage_owords];
};

template <int i>
struct StorageDude16
{
	template<uint ElementSize>
	__inline__ __device__ static StorageElement16<ElementSize>& assign(StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		StorageDude16<i - 1>::assign(dest, src);
		dest.storage[i] = src.storage[i];
		return dest;
	}

	template<uint ElementSize>
	__inline__ __device__ static StorageElement16<ElementSize>& load(StorageElement16<ElementSize>& dest, const volatile StorageElement16<ElementSize>& src)
	{
		StorageDude16<i - 1>::load(dest, src);
		::load(dest.storage[i], src.storage[i]);
		return dest;
	}

	template<uint ElementSize>
	__inline__ __device__ static volatile StorageElement16<ElementSize>& store(volatile StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		StorageDude16<i - 1>::store(dest, src);
		::store(dest.storage[i], src.storage[i]);
		return dest;
	}

	template<uint ElementSize>
	__inline__ __device__ static void clear(volatile StorageElement16<ElementSize>& dest)
	{
		StorageDude16<i - 1>::clear(dest);
		::clear(dest.storage[i]);
	}
};

template <>
struct StorageDude16<0>
{
	template<uint ElementSize>
	__inline__ __device__ static StorageElement16<ElementSize>& assign(StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		dest.storage[0] = src.storage[0];
		return dest;
	}

	template<uint ElementSize>
	__inline__ __device__ static StorageElement16<ElementSize>& load(StorageElement16<ElementSize>& dest, const volatile StorageElement16<ElementSize>& src)
	{
		::load(dest.storage[0], src.storage[0]);
		return dest;
	}

	template<uint ElementSize>
	__inline__ __device__ static volatile StorageElement16<ElementSize>& store(volatile StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
	{
		::store(dest.storage[0], src.storage[0]);
		return dest;
	}

	template<uint ElementSize>
	__inline__ __device__ static void clear(volatile StorageElement16<ElementSize>& dest)
	{
		::clear(dest.storage[0]);
	}
};


template<uint ElementSize>
__inline__ __device__ StorageElement16<ElementSize>& assign(StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::assign(dest, src);
}

template<uint ElementSize>
__inline__ __device__ StorageElement16<ElementSize>& load(StorageElement16<ElementSize>& dest, const volatile StorageElement16<ElementSize>& src)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::load(dest, src);
}

template<uint ElementSize>
__inline__ __device__ volatile StorageElement16<ElementSize>& store(volatile StorageElement16<ElementSize>& dest, const StorageElement16<ElementSize>& src)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::store(dest, src);
}

template<uint ElementSize>
__inline__ __device__ void clear(volatile StorageElement16<ElementSize>& dest)
{
	return StorageDude16<StorageElement16<ElementSize>::num_storage_owords - 1>::clear(dest);
}


struct StorageElement8
{
	uint2 storage;
};

__inline__ __device__ StorageElement8& assign(StorageElement8& dest, const StorageElement8& src)
{
	dest.storage = src.storage;
	return dest;
}

__inline__ __device__ StorageElement8& load(StorageElement8& dest, const volatile StorageElement8& src)
{
	load(dest.storage, src.storage);
	return dest;
}

__inline__ __device__ volatile StorageElement8& store(volatile StorageElement8& dest, const StorageElement8& src)
{
	store(dest.storage, src.storage);
	return dest;
}

__inline__ __device__ void clear(volatile StorageElement8& dest)
{
	clear(dest.storage);
}

template<uint TElementSize, bool take_eight>
struct StorageElementSelector
{
	typedef StorageElement16<TElementSize> type;
};

template<uint TElementSize>
struct StorageElementSelector<TElementSize, true>
{
	typedef StorageElement8 type;
};

template<uint TElementSize>
struct StorageElementTyping
{
  typedef typename StorageElementSelector<TElementSize, TElementSize <= 8>::type Type;
};

template<>
struct StorageElementTyping<0>;  // life finds a way...

template<>
struct StorageElementTyping<1>
{
  typedef unsigned char Type;
};
template<>
struct StorageElementTyping<2>
{
  typedef uchar2 Type;
};
template<>
struct StorageElementTyping<3>
{
  typedef uchar3 Type;
};
template<>
struct StorageElementTyping<4>
{
  typedef uint Type;
};



template <unsigned int width>
struct selectVectorCopyType;

template <>
struct selectVectorCopyType<16U>
{
	typedef uint4 type;
};

template <>
struct selectVectorCopyType<8U>
{
	typedef uint2 type;
};

template <>
struct selectVectorCopyType<4U>
{
	typedef uint1 type;
};

template <>
struct selectVectorCopyType<3U>
{
	typedef uchar3 type;
};

template <>
struct selectVectorCopyType<2U>
{
	typedef uchar2 type;
};

template <>
struct selectVectorCopyType<1U>
{
	typedef uchar1 type;
};

template <unsigned int bytes, int threads = 1>
struct vectorCopy
{
	static const unsigned int byte_width = bytes >= 16 ? 16 : bytes >= 8 ? 8 : bytes >= 4 ? 4 : 1;
	static const unsigned int iterations = bytes / byte_width;
	static const unsigned int max_threads = iterations < threads ? iterations : threads;
	static const unsigned int iterations_threaded = iterations / max_threads;
	static const unsigned int vectors_copied = max_threads * iterations_threaded;

	typedef typename selectVectorCopyType<byte_width>::type vector_type;

	__device__ __inline__ static void storeThreaded(volatile void* dest, const void* src, int i);
	__device__ __inline__ static void loadThreaded(void* dest, const volatile void* src, int i);
};

template <int threads>
struct vectorCopy<0, threads>
{
	__device__ __inline__ static void storeThreaded(volatile void* dest, const void* src, int i) {}
	__device__ __inline__ static void loadThreaded(void* dest, const volatile void* src, int i) {}
};

template <unsigned int bytes, int threads>
__device__ __inline__ void vectorCopy<bytes, threads>::storeThreaded(volatile void* dest, const void* src, int i)
{
	volatile vector_type* const destv = static_cast<volatile vector_type*>(dest);
	const vector_type* const srcv = static_cast<const vector_type*>(src);

	if (i < max_threads)
	{
		volatile vector_type* d = destv + i;
		const vector_type* s = srcv + i;
		#pragma unroll
		for (int j = 0; j < iterations_threaded; ++j)
		{
			store(*d, *s);
			d += max_threads;
			s += max_threads;
		}
	}

	vectorCopy<bytes - byte_width * vectors_copied, threads>::storeThreaded(destv + vectors_copied, srcv + vectors_copied, i);
}

template <unsigned int bytes, int threads>
__device__ __inline__ void vectorCopy<bytes, threads>::loadThreaded(void* dest, const volatile void* src, int i)
{
	vector_type* const destv = static_cast<volatile vector_type*>(dest);
	const volatile vector_type* const srcv = static_cast<const volatile vector_type*>(src);

	if (i < max_threads)
	{
		vector_type* d = destv + i;
		const volatile vector_type* s = srcv + i;
		#pragma unroll
		for (int j = 0; j < iterations_threaded; ++j)
		{
			load(*d, *s);
			d += max_threads;
			s += max_threads;
		}
	}

	vectorCopy<bytes - byte_width * vectors_copied, threads>::loadThreaded(destv + vectors_copied, srcv + vectors_copied, i);
}

template<int Threads, class T>
__device__ __inline__  void multiWrite(volatile T* data_out, T* data)
{
	vectorCopy<sizeof(T), Threads>::storeThreaded(data_out, data, Softshell::laneid() % Threads);

	//if (Softshell::laneid() % Threads == 0)
	//{
	//	for (int i = 0; i < sizeof(T); ++i)
	//		reinterpret_cast<volatile char*>(data_out)[i] = reinterpret_cast<char*>(data)[i];
	//}
}

template<int Threads, class T>
__device__ __inline__  void multiRead(T* data, volatile T* data_in)
{
	vectorCopy<sizeof(T), Threads>::loadThreaded(data, data_in, Softshell::laneid() % Threads);

	//if (Softshell::laneid() % Threads == 0)
	//{
	//	for (int i = 0; i < sizeof(T); ++i)
	//		reinterpret_cast<volatile char*>(data_in)[i] = reinterpret_cast<char*>(data)[i];
	//}
}


//__inline__ __device__ void readStorageElement(void* data, const volatile void* stored, uint size)
//{
  //uint* pData = reinterpret_cast<uint*>(data);
  //const volatile uint* pReadData =  reinterpret_cast<const volatile uint*>(stored); 

  //while(size >= 32)
  //{
  //  *reinterpret_cast<StorageElementTyping<32>::Type*>(pData) = 
  //    *reinterpret_cast<const volatile typename StorageElementTyping<32>::Type*>(pReadData);
  //  size -= 32;
  //  pData += 8; 
  //  pReadData += 8;
  //}
  //if(size >= 16)
  //{
  //  *reinterpret_cast<StorageElementTyping<16>::Type*>(pData) =
  //    *reinterpret_cast<const volatile typename StorageElementTyping<16>::Type*>(pReadData);
  //  size -= 16;
  //  pData += 4;
  //  pReadData += 4;
  //}
  //if(size >= 8)
  //{
  //  *reinterpret_cast<StorageElementTyping<8>::Type*>(pData) =
  //    *reinterpret_cast<const volatile typename StorageElementTyping<8>::Type*>(pReadData);
  //  size -= 8;
  //  pData += 2;
  //  pReadData += 2;
  //}
  //if(size > 0)
  //{
  //  *reinterpret_cast<StorageElementTyping<4>::Type*>(pData) =
  //    *reinterpret_cast<const volatile typename StorageElementTyping<4>::Type*>(pReadData);
  //}
//}

template<uint TElementSize, class TAdditionalData, uint TQueueSize>
class QueueStorage
{
protected:
  typedef typename StorageElementTyping<TElementSize>::Type QueueData_T;
  typedef typename StorageElementTyping<sizeof(TAdditionalData)>::Type QueueAddtionalData_T;
  QueueData_T volatile storage[TQueueSize];
  QueueAddtionalData_T volatile additionalStorage[TQueueSize];

public:

  static std::string name()
  {
    return "";
  }
  
  __inline__ __device__ void init()
  {
  }

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

    //storage[pos.x] = *reinterpret_cast<QueueData_T*>(&data);
    store(storage[pos.x], *reinterpret_cast<QueueData_T*>(&data));
    //additionalStorage[pos.x] = *reinterpret_cast<QueueAddtionalData_T*>(&additionalData);
	 store(additionalStorage[pos.x], *reinterpret_cast<QueueAddtionalData_T*>(&additionalData));
  }

  template<int TThreadsPerElenent, class T>
  __inline__ __device__ void writeDataParallel(T* data, TAdditionalData additionalData, uint2 pos)
  {
    pos.x = pos.x%TQueueSize;
    multiWrite<TThreadsPerElenent, T>(reinterpret_cast<volatile T*>(storage + pos.x), data);
    multiWrite<TThreadsPerElenent, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(additionalStorage + pos.x), &additionalData);

    ////TODO this could be unrolled in some cases...
    //for(int i = Softshell::laneid()%TThreadsPerElenent; i < TElementSize/sizeof(uint); i+=TThreadsPerElenent)
    //  reinterpret_cast<volatile uint*>(storage + pos.x)[i] = reinterpret_cast<uint*>(data)[i];

    //for(int i = Softshell::laneid()%TThreadsPerElenent; i < sizeof(TAdditionalData)/sizeof(uint); i+=TThreadsPerElenent)
    //  reinterpret_cast<volatile uint*>(additionalStorage + pos.x)[i] = reinterpret_cast<uint*>(&additionalData)[i];
  }

  __inline__ __device__ void readData(void* data, TAdditionalData* additionalData, uint pos)
  {
    pos = pos%TQueueSize;
    //*reinterpret_cast<QueueData_T*>(data) = storage[pos];
    load(*reinterpret_cast<QueueData_T*>(data), storage[pos]);
    //*reinterpret_cast<QueueAddtionalData_T*>(additionalData) = additionalStorage[pos];
	 load(*reinterpret_cast<QueueAddtionalData_T*>(additionalData), additionalStorage[pos]);
  }

  __inline__ __device__ void* readDataPointers(TAdditionalData* additionalData, uint pos)
  {
    pos = pos%TQueueSize;
    //*reinterpret_cast<QueueAddtionalData_T*>(additionalData) = additionalStorage[pos];
	 load(*reinterpret_cast<QueueAddtionalData_T*>(additionalData), additionalStorage[pos]);
    return (void*)(storage + pos);
  }

  __inline__ __device__ uint getPosFromPointer(void* data)
  {
    return (((unsigned char*) data) - ((unsigned char*)storage))/sizeof(QueueData_T);
  }


  __inline__ __device__ void* readDataPointer(uint pos)
  {
    pos = pos%TQueueSize;
    return (void*)(storage + pos);
  }

  __inline__ __device__ void writeAdditionalData(TAdditionalData additionalData, uint pos)
  {
    pos = pos %TQueueSize;
    additionalStorage[pos] = *reinterpret_cast<QueueAddtionalData_T*>(&additionalData);
  }

  template<int TThreadsPerElenent>
  __inline__ __device__ void writeAdditionalDataParallel(TAdditionalData additionalData, uint pos)
  {
    pos = pos%TQueueSize;
    multiWrite<TThreadsPerElenent, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(additionalStorage + pos), &additionalData);
  }

  __inline__ __device__ void storageFinishRead(uint2 pos)
  {
  }
};

template<uint TElementSize, uint TQueueSize>
class QueueStorage<TElementSize, void, TQueueSize>
{
protected:
  typedef typename StorageElementTyping<TElementSize>::Type QueueData_T;
  QueueData_T volatile storage[TQueueSize];

public:

  static std::string name()
  {
    return "";
  }

  __inline__ __device__ void init()
  {
  }

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
    //storage[pos.x] = *reinterpret_cast<QueueData_T*>(&data);
    store(storage[pos.x], *reinterpret_cast<QueueData_T*>(&data));
//printf("TQueueSize: %d, Elementsize %d, offset0: %llx, offset1 %llx\n", TQueueSize,TElementSize, &storage[0], &storage[1]);
  }

  template<int TThreadsPerElenent, class T>
  __inline__ __device__ void writeDataParallel(T* data, uint2 pos)
  {
    pos.x = pos.x%TQueueSize;
    multiWrite<TThreadsPerElenent, T>(reinterpret_cast<volatile T*>(storage + pos.x), data);

    ////TODO this could be unrolled in some cases...
    //for(int i = Softshell::laneid()%TThreadsPerElenent; i < TElementSize/sizeof(uint); i+=TThreadsPerElenent)
    //  reinterpret_cast<volatile uint*>(storage + pos.x)[i] = reinterpret_cast<uint*>(data)[i];
  }

  __inline__ __device__ void readData(void* data, uint pos)
  {
    pos = pos%TQueueSize;
    load(*reinterpret_cast<QueueData_T*>(data), storage[pos]);
  }

  __inline__ __device__ void* readDataPointers(uint pos)
  {
    pos = pos%TQueueSize;
    return (void*)(storage + pos);
  }

  __inline__ __device__ void* readDataPointer(uint pos)
  {
    pos = pos%TQueueSize;
    return (void*)(storage + pos);
  }

  __inline__ __device__ uint getPosFromPointer(void* data)
  {
    return (((unsigned char*) data) - ((unsigned char*)storage))/sizeof(QueueData_T);
  }

  __inline__ __device__ void storageFinishRead(uint2 pos)
  {
  }
};


template<uint TElementSize, uint TQueueSize, class TAdditionalData, class QueueStub, class TQueueStorage >
class QueueBuilder : public ::BasicQueue<TAdditionalData>, protected TQueueStorage, public QueueStub
{
  static const uint ElementSize = (TElementSize + sizeof(uint) - 1)/sizeof(uint);

public:
  static const bool supportReuseInit = false;

  __inline__ __device__ void init()
  {
    QueueStub::init();
    TQueueStorage::init();
  }

  static std::string name()
  {
    return QueueStub::name() + TQueueStorage::name();
  }

  template<int TThreadssPerElment, class Data>
  __inline__ __device__ bool enqueueInitial(Data* data, TAdditionalData additionalData)
  {
    return enqueue<TThreadssPerElment, Data>(data, additionalData);
  }

  template<class Data>
  __inline__ __device__ bool enqueueInitial(Data data, TAdditionalData additionalData) 
  {
    return enqueue<Data>(data, additionalData);
  }

  template<class Data>
  __device__ bool enqueue(Data data, TAdditionalData additionalData) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo = prepareData (data, additionalData);
    do
    {
      pos = QueueStub:: template enqueuePrep<1>(pos);
      if(pos.x >= 0)
      {
          writeData(data, additionalData, make_uint2(pos.x, addinfo) );
          __threadfence();
          QueueStub:: template enqueueEnd<1>(pos);
      }
    } while(pos.x == -2);
    return pos.x >= 0;
  }

  template<int TThreadssPerElment, class Data>
  __device__ bool enqueue(Data* data, TAdditionalData additionalData) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo =  TQueueStorage :: template prepareDataParallel<TThreadssPerElment> (data, additionalData);
    do
    {
      pos = QueueStub:: template enqueuePrep<TThreadssPerElment>(pos);
      if(pos.x >= 0)
      {
           TQueueStorage :: template writeDataParallel<TThreadssPerElment> (data, additionalData, make_uint2(pos.x, addinfo) );
          __threadfence();
          QueueStub:: template enqueueEnd<TThreadssPerElment>(pos);
      }
    } while(pos.x == -2);
    return pos.x >= 0;
  }

  __inline__ __device__ int dequeue(void* data, TAdditionalData* addtionalData, int num)
  {
      
    uint2 offset_take = QueueStub::dequeuePrep(num);

    if(threadIdx.x < offset_take.y)
    {
      readData(reinterpret_cast<uint*>(data) + threadIdx.x * ElementSize, addtionalData + threadIdx.x, offset_take.x + threadIdx.x);
      __threadfence();
    }
    __syncthreads();
    QueueStub::dequeueEnd(offset_take); 
    this->storageFinishRead(offset_take);
    return offset_take.y;
  }
};

template<uint TElementSize, uint TQueueSize, class QueueStub, class TQueueStorage >
class QueueBuilder<TElementSize, TQueueSize, void, QueueStub, TQueueStorage>
  : public ::BasicQueue<void>, protected TQueueStorage, public QueueStub
{
  static const uint ElementSize = (TElementSize + sizeof(uint) - 1)/sizeof(uint);
public:
  static const bool supportReuseInit = false;

  __inline__ __device__ void init()
  {
    QueueStub::init();
    TQueueStorage::init();
  }

  static std::string name()
  {
    return QueueStub::name() + TQueueStorage::name();
  }

  template<class Data>
  __inline__ __device__ bool enqueueInitial(Data data) 
  {
    return enqueue<Data>(data);
  }

  template<int TThreadssPerElment, class Data>
  __inline__ __device__ bool enqueueInitial(Data* data)
  {
    return enqueue<TThreadssPerElment, Data>(data);
  }

  template<class Data>
  __device__ bool enqueue(Data data) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo = prepareData(data);
    do
    {
      pos = QueueStub::template enqueuePrep<1>(pos);
      if(pos.x >= 0)
      {
        writeData(data, make_uint2(pos.x, addinfo));
        __threadfence();
        QueueStub:: template enqueueEnd<1>(pos);
      }
    } while(pos.x == -2);
    return pos.x >= 0;
  }

   template<int TThreadssPerElment, class Data>
  __device__ bool enqueue(Data* data) 
  {        
    int2 pos = make_int2(-1,0);
    uint addinfo =  TQueueStorage :: template prepareDataParallel<TThreadssPerElment> (data);
    do
    {
      pos = QueueStub:: template enqueuePrep<TThreadssPerElment>(pos);
      if(pos.x >= 0)
      {
           TQueueStorage :: template writeDataParallel<TThreadssPerElment> (data, make_uint2(pos.x, addinfo) );
          __threadfence();
          QueueStub:: template enqueueEnd<TThreadssPerElment>(pos);
      }
    } while(pos.x == -2);
    return pos.x >= 0;
  }

  __inline__ __device__ int dequeue(void* data, int num)
  {
    uint2 offset_take = QueueStub::dequeuePrep(num);
    if(threadIdx.x < offset_take.y)
    {
      this->readData(reinterpret_cast<uint*>(data) + threadIdx.x * ElementSize, offset_take.x + threadIdx.x);
      __threadfence();
    }
    __syncthreads();
    QueueStub::dequeueEnd(offset_take); 
    this->storageFinishRead(offset_take);
    return offset_take.y;
  }
};





//FIXME: class is not overflowsave / has no free!!!!
template<uint MemSize>
class MemoryAllocFastest
{
  static const uint AllocElements = MemSize/sizeof(uint);
  uint allocPointer;
public:
  uint4 volatile dataAllocation[AllocElements/4];

  __inline__ __device__ void init()
  {
    uint lid = threadIdx.x + blockIdx.x*blockDim.x;
    if(lid == 0)
      allocPointer = 0;
  }
  __inline__ __device__  uint allocOffset(uint size)
  {
    size = size/sizeof(uint);
    uint p = atomicAdd(&allocPointer,size)%AllocElements;
    while(p + size > AllocElements)
      p = atomicAdd(&allocPointer,size)%AllocElements;
    return  p;
  }

  __inline__ __device__ volatile uint* offsetToPointer(uint offset)
  {
    return  reinterpret_cast<volatile uint*>(dataAllocation) + offset;
  }
  __inline__ __device__ volatile uint* alloc(uint size)
  {
    return  offsetToPointer(allocOffset(size));
  }

  __inline__ __device__  void free(void *p, int size)
  {
  }
    __inline__ __device__  void freeOffset(int offset, int size)
  {
  }
};

//FIXME: allocator is only safe for elements with are a power of two mulitple of 16 bytes (or smaller than 16 bytes)
// and the multiple must be <= 32*16 bytes
template<uint MemSize>
class MemoryAlloc
{
  static const uint AllocSize = 16;
  static const uint AllocElements = MemSize/AllocSize;
  
  uint flags[(AllocElements + 31)/32];
  uint allocPointer;
public:
  uint4 volatile dataAllocation[AllocElements];

  __inline__ __device__ void init()
  {
    uint lid = threadIdx.x + blockIdx.x*blockDim.x;
    for(int i = lid; i < (AllocElements + 31)/32; i += blockDim.x*gridDim.x)
      flags[i] = 0;
    if(lid == 0)
      allocPointer = 0;
  }
  __inline__ __device__  int allocOffset(uint size)
  {
    size = (size+AllocSize-1)/AllocSize;
    for(uint t = 0; t < AllocElements/AllocSize; ++t)
    {
      int p = atomicAdd(&allocPointer,size)%AllocElements;
      if(p + size > AllocElements)
        p = atomicAdd(&allocPointer,size)%AllocElements;
      //check bits
      int bigoffset = p / 32;
      int withinoffset = p - bigoffset*32;
      uint bits = ((1u << size)-1u) << withinoffset;
      uint oldf = atomicOr(flags + bigoffset, bits);
      if((oldf & bits) == 0)
        return p;
      atomicAnd(flags + bigoffset, oldf | (~bits));
    }
    //printf("could not find a free spot!\n");
    return -1;
  }

  __inline__ __device__ volatile uint* offsetToPointer(int offset)
  {
    return  reinterpret_cast<volatile uint*>(dataAllocation + offset);
  }
  __inline__ __device__ int pointerToOffset(void *p)
  {
    return (reinterpret_cast<volatile uint4*>(p)-dataAllocation);
  }
  __inline__ __device__ volatile uint* alloc(uint size)
  {
    return  offsetToPointer(allocOffset(size));
  }

  __inline__ __device__  void free(void *p, int size)
  {
    freeOffset(pointerToOffset(p), size);
  }
    __inline__ __device__  void freeOffset(int offset, int size)
  {
    //printf("free called for %d %d\n",offset, size);
    size = (size+AllocSize-1)/AllocSize;
    int bigoffset = offset / 32;
    int withinoffset = offset - bigoffset*32;
    uint bits = ((1u << size)-1u) << withinoffset;
    atomicAnd(flags + bigoffset, ~bits);
  }
};

template<uint TAvgElementSize, class TAdditionalData, uint TQueueSize, bool TCheckSet = false, template<uint > class MemAlloc = MemoryAlloc>
class AllocStorage : private MemAlloc<TQueueSize*(TAvgElementSize + (TAvgElementSize > 8 || AdditionalDataInfo<TAdditionalData>::size > 8 ? (sizeof(TAdditionalData)+15)/16*16 :  TAvgElementSize > 4 || AdditionalDataInfo<TAdditionalData>::size > 4 ? (sizeof(TAdditionalData)+7)/8*8 : 4))>
{

protected:
  static const  int ForceSize = TAvgElementSize > 8 ? 16 :
                                TAvgElementSize > 4 ? 8 : 4;
  static const  int PureAdditionalSize = (sizeof(TAdditionalData)+sizeof(uint)-1)/sizeof(uint);
  static const  int AdditionalSize = TAvgElementSize > 8 || sizeof(TAdditionalData) > 8 ? (sizeof(TAdditionalData)+15)/16*16 :
                                     TAvgElementSize > 4 || sizeof(TAdditionalData) > 4 ? (sizeof(TAdditionalData)+7)/8*8 : 4;

  typedef typename StorageElementTyping<sizeof(TAdditionalData)>::Type AdditonalInfoElement;
  typedef typename StorageElementTyping<sizeof(uint2)>::Type OffsetData_T;

  OffsetData_T volatile offsetStorage[TQueueSize];

public:

  static std::string name()
  {
    return std::string("Alloced");// + std::to_string((unsigned long long)AdditionalSize) + " " + std::to_string((unsigned long long)TAvgElementSize);
  }
  
  __inline__ __device__ void init()
  {
    MemAlloc<TQueueSize*(TAvgElementSize + (TAvgElementSize > 8 || AdditionalDataInfo<TAdditionalData>::size > 8 ? (sizeof(TAdditionalData)+15)/16*16 :  TAvgElementSize > 4 || AdditionalDataInfo<TAdditionalData>::size > 4 ? (sizeof(TAdditionalData)+7)/8*8 : 4))>::init();
    if(TCheckSet)
    {
       uint lid = threadIdx.x + blockIdx.x*blockDim.x;
       for(uint i = lid; i < 2*TQueueSize; i+=blockDim.x*gridDim.x)
         ((uint*)offsetStorage)[i] = 0;
    }
  }

  template<class T>
  __inline__ __device__ uint prepareData(T data, TAdditionalData additionalData)
  {
    uint p = allocOffset((sizeof(T) + AdditionalSize + ForceSize - 1)/ForceSize*ForceSize);
    *reinterpret_cast<volatile AdditonalInfoElement*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + p) = *reinterpret_cast<AdditonalInfoElement*>(&additionalData);
    *reinterpret_cast<volatile typename StorageElementTyping<sizeof(T)>::Type*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + p + AdditionalSize/sizeof(uint) ) = *reinterpret_cast<typename StorageElementTyping<sizeof(T)>::Type*>(&data);
    return p;
  }

  template<int TThreadsPerElement, class T>
  __inline__ __device__ uint prepareDataParallel(T* data, TAdditionalData additionalData)
  {
    if(TThreadsPerElement == 1)
      return prepareData(*data, additionalData);

    int p;
    if(Softshell::laneid()%TThreadsPerElement == 0)
      p = allocOffset((sizeof(T) + AdditionalSize + ForceSize - 1)/ForceSize*ForceSize);
    p = warpBroadcast<TThreadsPerElement>(p, 0);
    //p = __shfl(p, 0, TThreadsPerElement);    
    multiWrite<TThreadsPerElement, TAdditionalData>(reinterpret_cast<volatile TAdditionalData*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + p), &additionalData);
    multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + p + AdditionalSize/sizeof(uint)), data);

    return p;
  }

  template<class T>
  __inline__ __device__ void writeData(T data, TAdditionalData additionalData, uint2 pos)
  {
    pos.x = pos.x%TQueueSize;
    uint2 o = make_uint2(pos.y, sizeof(T));

    if(TCheckSet)
    {
      o.x += 1;
      while(*(((volatile uint*)offsetStorage) + 2*pos.x) != 0)
        __threadfence();
    }

    offsetStorage[pos.x] = *reinterpret_cast<OffsetData_T*>(&o);
  }

  template<int TThreadsPerElement,class T>
  __inline__ __device__ void writeDataParallel(T* data, TAdditionalData additionalData, uint2 pos)
  {
    if(Softshell::laneid()%TThreadsPerElement == 0)
      writeData(*data,  additionalData, pos);
  }

  __inline__ __device__ void readData(void* data, TAdditionalData* additionalData, uint pos)
  {
    OffsetData_T offsetData;
    pos = pos%TQueueSize;
    offsetData  = offsetStorage[pos];
    uint2 offset = *reinterpret_cast<uint2*>(&offsetData);

    if(TCheckSet)
    {
      while( offset.x == 0 || offset.y == 0)
      {
        __threadfence();
        offsetData  = offsetStorage[pos];
        offset = *reinterpret_cast<uint2*>(&offsetData);
      }
      offset.x -= 1;
    }
    
    *reinterpret_cast<AdditonalInfoElement*>(additionalData) = *reinterpret_cast<volatile AdditonalInfoElement*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + offset.x);
    this->readStorageElement(data, reinterpret_cast<volatile uint*>(this->dataAllocation) + offset.x + AdditionalSize/sizeof(uint), offset.y);
   
  }
  __inline__ __device__ void storageFinishRead(uint2 pos)
  {
     
    if(threadIdx.x < pos.y)
    {
      uint p = (pos.x + threadIdx.x) % TQueueSize;

      OffsetData_T offsetData;
      offsetData  = offsetStorage[p];
      uint2 offset = *reinterpret_cast<uint2*>(&offsetData);

      this->freeOffset(offset.x, offset.y);
      if(TCheckSet)
      {
        __threadfence();
        uint2 o = make_uint2(0, 0);
        offsetStorage[p] = *reinterpret_cast<OffsetData_T*>(&o);
      }
    }
  }
};

template<uint TAvgElementSize, uint TQueueSize, bool TCheckSet, template<uint > class MemAlloc>
class AllocStorage<TAvgElementSize, void, TQueueSize, TCheckSet, MemAlloc> : private MemAlloc<TAvgElementSize*TQueueSize>
{
protected:
  static const  int ForceSize = TAvgElementSize > 8 ? 16 :
                                TAvgElementSize > 4 ? 8 : 4;
  
  typedef typename StorageElementTyping<sizeof(uint2)>::Type OffsetData_T;

  OffsetData_T volatile offsetStorage[TQueueSize];

public:

  static std::string name()
  {
    return "Alloced";
  }
  
  __inline__ __device__ void init()
  {
    MemAlloc<TAvgElementSize*TQueueSize>::init();
    if(TCheckSet)
    {
       uint lid = threadIdx.x + blockIdx.x*blockDim.x;
       for(uint i = lid; i < 2*TQueueSize; i+=blockDim.x*gridDim.x)
         ((uint*)offsetStorage)[i] = 0;
    }
  }

  template<class T>
  __inline__ __device__ uint prepareData(T data)
  {
    uint p = allocOffset((sizeof(T) + ForceSize - 1)/ForceSize*ForceSize);
    *reinterpret_cast<volatile typename StorageElementTyping<sizeof(T)>::Type*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + p ) = *reinterpret_cast<typename StorageElementTyping<sizeof(T)>::Type*>(&data);
    return p;
  }

  template<int TThreadsPerElement, class T>
  __inline__ __device__ uint prepareDataParallel(T* data)
  {
    if(TThreadsPerElement == 1)
      return prepareData(*data);

    int p;
    if(Softshell::laneid()%TThreadsPerElement == 0)
      p = allocOffset((sizeof(T) + ForceSize - 1)/ForceSize*ForceSize);
    //p = __shfl(p, 0, TThreadsPerElement);
    p = warpBroadcast<TThreadsPerElement>(p, 0);
    multiWrite<TThreadsPerElement, T>(reinterpret_cast<volatile T*>(reinterpret_cast<volatile uint*>(this->dataAllocation) + p), data);
    return p;
  }

  template<class T>
  __inline__ __device__ void writeData(T data, uint2 pos)
  {
    pos.x = pos.x%TQueueSize;    
    uint2 o = make_uint2(pos.y, sizeof(T));

    if(TCheckSet)
    {
      o.x += 1;
      while(*(((volatile uint*)offsetStorage) + 2*pos.x) != 0)
        __threadfence();
    }

    offsetStorage[pos.x] =  *reinterpret_cast<OffsetData_T*>(&o);
  }

  template<int TThreadsPerElement, class T>
  __inline__ __device__ void writeDataParallel(T* data, uint2 pos)
  {
    if(Softshell::laneid()%TThreadsPerElement == 0)
      writeData(*data, pos);
  }

  __inline__ __device__ void readData(void* data, uint pos)
  {
    OffsetData_T offsetData;
    pos = pos%TQueueSize;  
    offsetData  = offsetStorage[pos];
    uint2 offset = *reinterpret_cast<uint2*>(&offsetData);

    if(TCheckSet)
    {
      while( offset.x == 0 || offset.y == 0)
      {
        __threadfence();
        offsetData  = offsetStorage[pos];
        offset = *reinterpret_cast<uint2*>(&offsetData);
      }
      offset.x -= 1;
    }
    
    this->readStorageElement(data, reinterpret_cast<volatile uint*>(this->dataAllocation) + offset.x, offset.y);
  }
  __inline__ __device__ void storageFinishRead(uint2 pos)
  {
     if(threadIdx.x < pos.y)
    {
      uint p = (pos.x + threadIdx.x) % TQueueSize;
      OffsetData_T offsetData;
      offsetData  = offsetStorage[p];
      uint2 offset = *reinterpret_cast<uint2*>(&offsetData);

      this->freeOffset(offset.x, offset.y);
      if(TCheckSet)
      {
        __threadfence();
        uint2 o = make_uint2(0, 0);
        offsetStorage[p] = *reinterpret_cast<OffsetData_T*>(&o);
      }
    }
  }
};

 
