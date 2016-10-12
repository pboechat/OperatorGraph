#pragma once
#include "queueInterface.cuh"
#include "tools/common.cuh"
#include "queueHelpers.cuh"



template<uint TElementSize, class TAdditionalData, class MemPool>
class QueueDistLocksWithPool;


template<uint MemPoolSize, uint TPageSize = 32U*1024U>
class MemPool
{
public:
	static const uint PageSize = TPageSize;
private:
	static const uint NumPages = (MemPoolSize - 12) / (PageSize + 4);
	static const uint Empty = 0xFFFFFFFF;
	struct Page
	{
		char data[PageSize];
	};
	
	Page pages[NumPages];

	volatile uint availablePages[NumPages];
	uint front, back;
	int count;
public:

	__device__ void init()
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid == 0)
		{ 
			front = back = 0;
			count = NumPages;
		}
		for (int i = tid; i < PageSize * NumPages / 4; i += blockDim.x*gridDim.x)
			reinterpret_cast<uint*>(pages)[i] = 0U;
		for (int i = tid; i < NumPages; i += blockDim.x*gridDim.x)
			availablePages[i] = i;
	}
	__device__ void* getPage()
	{
		int n = atomicSub(&count, 1);
		if (n <= 0)
		{
			atomicAdd(&count, 1);
			return nullptr;
		}
		uint pos = atomicAdd(&front, 1) % NumPages;
		uint page;
		while ((page = availablePages[pos]) == Empty)
			__threadfence();
		//printf("using page %d\n", page);
		availablePages[pos] = Empty;
		return pages + page;
	}
	__device__ void releasePage(void* page)
	{
		atomicAdd(&count, 1);
		uint pid = reinterpret_cast<Page*>(page) - pages;
		//printf("freeing page %d\n", pid);
		uint pos = atomicAdd(&back, 1) % NumPages;
		//TODO: should be able to remove this check here as we cannot have more pages here anyway than we can hold...
		while (availablePages[pos] != Empty)
			__threadfence();
		availablePages[pos] = pid;
	}
};


__constant__ void* qdl_mempool;

template<class MemPool>
__global__ void initMemPoolD()
{
	//if (threadIdx.x == 0 && blockIdx.x == 0)
	//	printf("d pool: %p\n", qdl_mempool);
	reinterpret_cast<MemPool*>(qdl_mempool)->init();
}


typedef void(*Mempoolinitfunc)();
Mempoolinitfunc MemPoolInitializer = nullptr;

template<class MemPool>
void initMemPoolInstance()
{
	initMemPoolD<MemPool> <<<512, 512>>>();
}

template<class MemPool>
void setMemPool(MemPool* dpool)
{
	CUDA_CHECKED_CALL(cudaMemcpyToSymbol(qdl_mempool, &dpool, sizeof(void*)));
	MemPoolInitializer = initMemPoolInstance < MemPool > ;
	initMemPoolInstance < MemPool >();
}


void initMemPool()
{
	if (MemPoolInitializer)
		MemPoolInitializer();
}


template<uint TElementSize, class MemPool>
class QueueDistLocksWithPool<TElementSize, void, MemPool> : public BasicQueue<void>
{
	typedef typename StorageElementTyping<TElementSize>::Type QueueDataType;
	static const int PageElements = (MemPool::PageSize - 2 * sizeof(void *) - 16) / (sizeof(QueueDataType) + 4);

	struct Page
	{
		volatile QueueDataType data[PageElements];
		uint locks[PageElements];
		Page *next, *prev;
		uint back;
		uint pageOffset;
		uint doneCount;
	};


	__device__ static MemPool* getPool() 
	{
		return reinterpret_cast <MemPool*> (qdl_mempool); 
	}

	Page *frontPage, *backPage;
	int count;
	uint front;


	__inline__ __device__ uint reserveElement(Page*& page)
	{
		atomicAdd(&count, 1);

		// get back page
		Page* back = backPage;
		uint p;

		while (true)
		{
			// try alloc on that page
			p = atomicAdd(&back->back, 1);
			if (p < PageElements)
				break;
			Page* nback;
			while ((nback = back->next) == nullptr)
				__threadfence();
			back = nback;
		}

		// if this is the first thread on that page, I have to add another one
		if (p == 0)
		{
			backPage = back;
			Page *npage = reinterpret_cast<Page*>(getPool()->getPage());
			if (npage == nullptr)
			{
				/*printf("DistLocks Mempool out of memory!");
				__threadfence();*/
				__trap();
			}
			else
			{
				npage->next = nullptr;
				npage->back = 0;
				npage->doneCount = 0;
				npage->pageOffset = back->pageOffset + PageElements;
				npage->prev = back;
				__threadfence();
				back->next = npage;
			}
		}

		page = back;
		return p;
	}

	__inline__ __device__ void beginRead(Page*& page_, int& eloffset, QueueDataType volatile *& data, QueueDataType volatile *& firstdata, int num, int pos)
	{
		__shared__ int offset;
		__shared__ Page * page;

		if (threadIdx.x == 0)
		{
			uint foffset = atomicAdd(&front, num);
			Page * tpage = frontPage;
			uint withinpageoffset;
			while ((withinpageoffset = foffset - tpage->pageOffset) > PageElements)
			{
				Page * npage;
				while ((npage = tpage->next) == nullptr)
					__threadfence();
				tpage = npage;
			}
			page = tpage;
			offset = withinpageoffset;
		}
		__syncthreads();

		firstdata = page->data + offset;

		if (pos < num)
		{ 
			Page * mpage = page;
			uint moffset = offset + pos;
			while (moffset >= PageElements)
			{ 
				Page* npage;
				while ((npage = mpage->next) == nullptr)
					__threadfence();
				mpage = npage;
				moffset -= PageElements;
			}

			//if (mpage == 0)
			//	printf("wat n0 f00k 1 %d + %d  num %d / pageleemtns %d\n", offset, pos, num, PageElements);

			// we just walked over at least one page
			if (moffset == 0 && num - pos < PageElements)
				frontPage = mpage;

			while (mpage->locks[moffset] != 1)
				__threadfence();
			data = mpage->data + moffset;
			page_ = mpage;
			eloffset = moffset;
		}
	}
	
	__inline__ __device__ void endRead(Page* page, int offset)
	{
		//if (page == 0 || page == reinterpret_cast<Page*>(0xc00000000ULL))
		//	printf("wat n0 f00k 2\n");
		// zero data
		clear(page->data[offset]);
		page->locks[offset] = 0;

		__threadfence();
		uint c = atomicAdd(&page->doneCount, 1);

		
		// remove possible old page
		if (c == PageElements - 1)
			getPool()->releasePage(page);
	}

public:
	__inline__ __device__ void init()
	{
		if (blockIdx.x == 0 && threadIdx.x == 0)
		{ 
			count = 0;
			front = 0U;
			frontPage = backPage = reinterpret_cast<Page*>(getPool()->getPage());
		}
	}

	template<class Data>
	__inline__ __device__ bool enqueueInitial(Data data)
	{
		return enqueue<Data>(data);
	}

	template<int threads, class Data>
	__inline__ __device__ bool enqueueInitial(Data* data)
	{
		return enqueue<threads, Data>(data);
	}

	template<class Data>
	__inline__ __device__ bool enqueue(Data data)
	{
		Page* page;
		uint pos = reserveElement(page);

		// store element
		store(page->data[pos],*reinterpret_cast<QueueDataType*>(&data));
		__threadfence();
		page->locks[pos] = 1;
		return true;
	}

	template<int threads, class Data>
	__inline__ __device__ bool enqueue(Data* data)
	{
		Page* page;
		uint pos;
		if (Softshell::laneid() % threads == 0)
			pos = reserveElement(page);

		//dist info
		//pos = warpBroadcast<threads>(pos, 0);
		//page = warpBroadcast<threads>(page, 0);
		//// store element
		//multiWrite<threads, QueueDataType>(page->data + pos, reinterpret_cast<QueueDataType*>(data));

		volatile QueueDataType* storelocation = warpBroadcast<threads>(page->data + pos, 0);
		multiWrite<threads, QueueDataType>(storelocation, reinterpret_cast<QueueDataType*>(data));

		__threadfence();

		if (Softshell::laneid() % threads == 0)
			page->locks[pos] = 1;
		return true;
	}

	// TODO

	//template<class Data>
	//__inline__ __device__ Data* reserveSpot()
	//{
	//	Page* page;
	//	uint pos = reserveElement(page);
	//	return  const_cast<Data*>(reinterpret_cast<volatile Data*>(page->data + pos));
	//}

	//template<int threads, class Data>
	//__inline__ __device__ Data* reserveSpot()
	//{
	//	Page* page;
	//	uint pos;
	//	if (Softshell::laneid() % threads == 0)
	//		pos = reserveElement(page);
	//	//dist info
	//	pos = warpBroadcast<threads>(pos, 0);
	//	page = warpBroadcast<threads>(page, 0);

	//	return reinterpret_cast<Data*>(page->data[p]);
	//}



	//template<class Data>
	//__inline__ __device__ void completeSpot(Data* spot)
	//{

	//}

	//template<int threads, class Data>
	//__inline__ __device__ void completeSpot(Data* spot)
	//{

	//}

	__inline__ __device__ int dequeue(void* data, int maxnum)
	{
		__shared__ int num;
		if (threadIdx.x == 0)
		{
			int c = atomicSub(const_cast<int*>(&count), maxnum);
			if (c < maxnum)
			{
				atomicAdd(const_cast<int*>(&count), min(maxnum, maxnum - c));
				num = max(c, 0);
			}
			else
				num = maxnum;
		}
		
		__syncthreads();

		
		Page* page;
		int offset;
		QueueDataType volatile *el, volatile *firstel;
		beginRead(page, offset, el, firstel, num, threadIdx.x);

		if (threadIdx.x < num)
		{
			// read the data
			__threafence();
			load(*(reinterpret_cast<QueueDataType>(data) + threadIdx.x), el);
			__threafence();

			endRead(page, offset);
		}
		return num;
	}

	__inline__ __device__ int reserveRead(int maxnum, bool only_read_all = false)
	{
		__shared__ int num;
		if (threadIdx.x == 0)
		{
			int c = atomicSub(const_cast<int*>(&count), maxnum);
			if (c < maxnum)
			{
				if (only_read_all)
				{
					atomicAdd(const_cast<int*>(&count), maxnum);
					num = 0;
				}
				else
				{
					atomicAdd(const_cast<int*>(&count), min(maxnum, maxnum - c));
					num = max(c, 0);
				}
			}
			else
				num = maxnum;
		}
		__syncthreads();
		return num;
	}
	__inline__ __device__ int startRead(void*& data, int pos, int num)
	{
		if (num <= 0)
			return 0;
		
		Page* p;
		int off;
		volatile QueueDataType *el;
		volatile QueueDataType *firstel;
		beginRead(p, off, el, firstel, num, pos);
		__threadfence();
		data = const_cast<QueueDataType*>(el);

		// encode first offset
		uint* fid = reinterpret_cast<uint*>(const_cast<QueueDataType*>(firstel));
		int offsetid = fid - reinterpret_cast<uint*>(qdl_mempool);
		return offsetid;
	}
	__inline__ __device__ void finishRead(int id, int num)
	{
		// clear slot
		if (threadIdx.x < num)
		{
			uint pageid = (id * 4) / MemPool::PageSize; 
			Page* page = reinterpret_cast<Page*>(reinterpret_cast<char*>(qdl_mempool)+pageid * MemPool::PageSize);
			uint offset = ((id * 4) - pageid * MemPool::PageSize) / sizeof(QueueDataType) + threadIdx.x;
			while (offset >= PageElements)
				page = page->next,
				offset -= PageElements;
			
			endRead(page, offset);
		}
	}


	static std::string name()
	{
		return "DistLocksMemPool";
	}

	__inline__ __device__ int size() const
	{
		return max(0,*const_cast<volatile int*>(&count));
	}

};


template<class MemPool>
struct MemPoolDistLocksQueueTyping
{
	template<uint TElementSize, uint TQueueSize, class TAdditionalData>
	class Type : public QueueDistLocksWithPool < TElementSize,  TAdditionalData, MemPool > {};
};