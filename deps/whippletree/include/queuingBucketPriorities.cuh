#pragma once
#include "queueInterface.cuh"
#include "common.cuh"
#include "random.cuh"


template<class PInfo, int TNumQueues>
class BucketLinearPriority
{
protected:
	static const int MinPriority = PInfo:: template Priority<0> ::MinPriority;
	static const int MaxPriority = PInfo:: template Priority<0> ::MaxPriority;
	static const int PriorityRange = MaxPriority - MinPriority;
public:
	static const int NumQueues = PriorityRange>0 ? (PriorityRange>TNumQueues ? TNumQueues : PriorityRange) : 1;

	template<class Procedure>
	__inline__ __device__ static unsigned int computePrioritySlot(typename Procedure::ExpectedData* data)
	{
		if (PriorityRange == 0)
			return 0;
		int pid = findProcId<PInfo, Procedure>::value;
		int priority = PInfo:: template Priority<0>::eval(pid, reinterpret_cast<void*>(data));

		int slot = NumQueues - 1 - max(0, min((priority - MinPriority)*NumQueues / PriorityRange, NumQueues - 1));
		return slot;
	}

	template<class Q>
	__inline__ __device__ static int computeDequeueOrder(Q* qs)
	{
		return 0;
	}

	__inline__ __device__ static int getNextDequeueQueue(const int & computeOrder, int attempt, bool & hasNext)
	{
		hasNext = attempt + 1 < NumQueues;
		return attempt;
	}

	static __host__ std::string spec()
	{
		return std::string("");
	}
};

template<class PInfo, int TNumQueues>
class BucketLinearPriorityFast : public BucketLinearPriority<PInfo, TNumQueues>
{
	__inline__ __device__ static int* sharedSpace()
	{
		__shared__ int loc_data[NumQueues + 2];
		return loc_data;
	}
public:
	
	template<class Q>
	__inline__ __device__ static int computeDequeueOrder(Q queues[NumQueues])
	{
		int* loc_data = sharedSpace();
		for (int i = threadIdx.x; i < NumQueues; i += blockDim.x)
			loc_data[i + 1] = queues[i].size() > 0 ? i : NumQueues;

			//min prefix sum
		int offset = 1;
		for (int d = NumQueues; d > 1; d = (d + 1) / 2)
		{
			__syncthreads();
			int ai = offset*(2 * threadIdx.x) + 1;
			int bi = offset*(2 * threadIdx.x + 1) + 1;
			if (bi <= NumQueues)
				loc_data[ai] = min(loc_data[ai], loc_data[bi]);
			offset *= 2;
		}
		if (threadIdx.x == 0)
			loc_data[0] = loc_data[1],
			loc_data[1] = NumQueues;

		for (int d = 1; d < NumQueues; d *= 2)
		{
			offset /= 2;
			int ai = offset*(2 * threadIdx.x) + 1;
			int bi = offset*(2 * threadIdx.x + 1) + 1;
			__syncthreads();
			if (bi <= NumQueues)
			{
				int t = loc_data[bi];
				loc_data[bi] = loc_data[ai];
				loc_data[ai] = min(loc_data[ai], t);
			}
		}
		__syncthreads();
		return min(loc_data[0], NumQueues - 1);
	}

	__inline__ __device__ static int getNextDequeueQueue(int & take, int attempt, bool & hasNext)
	{
		int qId = take;
		take = sharedSpace()[take + 1];
		hasNext = take < NumQueues;
		return qId;
	}

	static __host__ std::string spec()
	{
		return std::string("Opt");
	}
};

template<class PInfo, int TNumQueues>
class BucketProportionalChoice : public BucketLinearPriority<PInfo, TNumQueues>
{
public:

	template<class Q>
	__inline__ __device__ static int computeDequeueOrder(Q* qs)
	{
		__syncthreads();
		__shared__ int qStart;
		if (threadIdx.x == 0)
		{
			int r = qrandom::rand();
			qStart = (r * r / qrandom::Range * NumQueues / qrandom::Range + r * NumQueues / qrandom::Range) / 2;
		}
		__syncthreads();
		return qStart;
	}

	__inline__ __device__ static int getNextDequeueQueue(const int & qStart, int attempt, bool & hasNext)
	{
		int qId = (qStart + attempt) % NumQueues;

		hasNext = attempt + 1 < NumQueues;
		return qId;
	}
};

template<unsigned int WrapAroundTime, class TimeBasis>
struct BucketEarliestDeadlineFirstTyping
{
	template<class PInfo, int TNumQueues>
	class Type
	{
	public:
		static const int NumQueues = TNumQueues;

		template<class Procedure>
		__inline__ __device__ static unsigned int computePrioritySlot(typename Procedure::ExpectedData* data)
		{
			int pid = findProcId<PInfo, Procedure>::value;
			uint dealineTime = PInfo:: template Priority<0>::eval(pid, reinterpret_cast<void*>(data));
			uint slot = (dealineTime % WrapAroundTime) * NumQueues / WrapAroundTime;
			return slot;
		}

		template<class Q>
		__inline__ __device__ static unsigned int computeDequeueOrder(Q* qs)
		{
			return (TimeBasis::time() % WrapAroundTime) * NumQueues / WrapAroundTime;
		}

		__inline__ __device__ static int getNextDequeueQueue(unsigned int queueOffset, int attempt, bool & hasNext)
		{
			hasNext = (attempt + 1) < NumQueues;
			return (queueOffset + attempt) % NumQueues;
		}
	};

	template<class PInfo, int TNumQueues>
	class FastType : public Type<PInfo, TNumQueues>
	{
		__inline__ __device__ static int* sharedSpace()
		{
			__shared__ int loc_data[NumQueues + 2];
			return loc_data;
		}
	public:

		template<class Q>
		__inline__ __device__ static int computeDequeueOrder(Q queues[NumQueues])
		{
			__syncthreads();
			int* loc_data = sharedSpace();
			loc_data[NumQueues + 1] = (TimeBasis::time() % WrapAroundTime) * NumQueues / WrapAroundTime;
			__syncthreads();


			for (int i = threadIdx.x; i < NumQueues; i += blockDim.x)
				loc_data[i + 1] = queues[(loc_data[NumQueues + 1] + i) % NumQueues].size() > 0 ? i : NumQueues;

			//min prefix sum
			int offset = 1;
			for (int d = NumQueues; d > 1; d = (d + 1) / 2)
			{
				__syncthreads();
				int ai = offset*(2 * threadIdx.x) + 1;
				int bi = offset*(2 * threadIdx.x + 1) + 1;
				if (bi <= NumQueues)
					loc_data[ai] = min(loc_data[ai], loc_data[bi]);
				offset *= 2;
			}
			if (threadIdx.x == 0)
				loc_data[0] = loc_data[1],
				loc_data[1] = NumQueues;

			for (int d = 1; d < NumQueues; d *= 2)
			{
				offset /= 2;
				int ai = offset*(2 * threadIdx.x) + 1;
				int bi = offset*(2 * threadIdx.x + 1) + 1;
				__syncthreads();
				if (bi <= NumQueues)
				{
					int t = loc_data[bi];
					loc_data[bi] = loc_data[ai];
					loc_data[ai] = min(loc_data[ai], t);
				}
			}
			__syncthreads();
			return loc_data[0];
		}

		__inline__ __device__ static int getNextDequeueQueue(int & take, int attempt, bool & hasNext)
		{
			int qId = take;
			take = sharedSpace()[take + 1];
			hasNext = take < NumQueues;
			return (sharedSpace()[NumQueues + 1] + qId) % NumQueues;
		}
	};
};

template<class PInfo, template<class /*ProcedureInfo*/> class TInternalQueueing, unsigned int TNumQueues, template<class /*ProcInfo*/, int /*NumQueues*/> class TQueueChooser = BucketLinearPriority>
class BucketPriorityQueue : public ::Queue<> 
{
protected:
	
	typedef TQueueChooser<PInfo, TNumQueues> QueueChooser;
	static const int NumQueues = QueueChooser::NumQueues;
	typedef TInternalQueueing<PInfo> InternalQueue;

	InternalQueue queues[NumQueues];
	 
	int dummy[128]; //compiler alignment mismatch hack

public:

	static const bool supportReuseInit = InternalQueue::supportReuseInit;

	static std::string name()
	{
		return std::string("BucketPriority") + QueueChooser::spec() + std::string("_") + std::to_string((long long)TNumQueues) + "_" + InternalQueue::name();
	}

	__inline__ __device__ void init() 
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].init();
	}

	template<class PROCEDURE>
	__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data) 
	{
		unsigned int qId = QueueChooser:: template computePrioritySlot<PROCEDURE>(&data);
		return queues[qId] . template enqueueInitial<PROCEDURE>(data);
	}

	template<int threads, class PROCEDURE>
	__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData* data)
	{
		unsigned int qId = QueueChooser:: template computePrioritySlot<PROCEDURE>(data);
		return queues[qId] . template enqueueInitial<threads, PROCEDURE>(data);
	}

	template<class PROCEDURE>
	__inline__ __device__  bool enqueue(typename PROCEDURE::ExpectedData data) 
	{
		unsigned int qId = QueueChooser:: template computePrioritySlot<PROCEDURE>(&data);
		return queues[qId] . template enqueue<PROCEDURE>(data);
	}

	template<int threads, class PROCEDURE>
	__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data) 
	{
		unsigned int qId = QueueChooser:: template computePrioritySlot<PROCEDURE>(data);
		return queues[qId] . template enqueue<threads,PROCEDURE>(data);
	}

	template<bool MultiProcedure>
	__inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = 100000)
	{
		auto orderInfo = QueueChooser::computeDequeueOrder(queues);

		bool hasNext = true;
		for(int attempt = 0; hasNext; ++attempt)
		{
			int qId = QueueChooser::getNextDequeueQueue(orderInfo, attempt, hasNext);
			int num = queues[qId] . template dequeue<MultiProcedure>(data, procId, maxShared);
			if (num != 0)
				return num;
			__syncthreads();
		}
		return 0;
	}

	template<bool MultiProcedure>
	__inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
	{
		auto orderInfo = QueueChooser::computeDequeueOrder(queues);

		bool hasNext = true;
		for (int attempt = 0; hasNext; ++attempt)
		{
			int qId = QueueChooser::getNextDequeueQueue(orderInfo, attempt, hasNext);
			int num = queues[qId] . template dequeueSelected<MultiProcedure>(data, procId, maxNum);
			if (num != 0)
				return num;
			__syncthreads();
		}
		return 0;
	}

	template<bool MultiProcedure>
	__inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = 100000)
	{ 
		auto orderInfo = QueueChooser::computeDequeueOrder(queues);

		bool hasNext = true;
		for (int attempt = 0; hasNext; ++attempt)
		{
			int qId = QueueChooser::getNextDequeueQueue(orderInfo, attempt, hasNext);
			int num = queues[qId] . template dequeueStartRead<MultiProcedure>(data, procId, maxShared);
			if (num != 0)
			{
				__syncthreads();
				procId[1] = (procId[1] & 0xFFFFFF) | (qId << 24);
				return num;
			}
			__syncthreads();
		}
		return 0;
	}

	template<class PROCEDURE>
	__inline__ __device__ void finishRead(int id,  int num)
	{
		int qId = id >> 24;
		return queues[qId] . template finishRead<PROCEDURE>(id & 0xFFFFFF, num);
	}
  
	__inline__ __device__ void workerStart()
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].workerStart();
	}
	__inline__ __device__ void workerMaintain()
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].workerMaintain();
	}
	__inline__ __device__ void workerEnd()
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].workerEnd();
	}
	__inline__ __device__ void globalMaintain()
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].globalMaintain();
	}

	__inline__ __device__ void record()
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].record();
	}

	__inline__ __device__ void reset()
	{
		for(int i = 0; i < NumQueues; ++i)
			queues[i].reset();
	}

	__inline__ __device__ void clear()
	{
		for (int i = 0; i < NumQueues; ++i)
			queues[i].clear();
	}
};


template<template<class ProcedureInfo> class TInternalQueueing, unsigned int TNumQueues, template<class /*ProcInfo*/, int /*NumQueues*/> class TQueueChooser = BucketLinearPriority>
struct BucketPriorityQueueTyping
{
	template<class ProcedureInfo>
	class Type : public BucketPriorityQueue<ProcedureInfo, TInternalQueueing, TNumQueues, TQueueChooser> {};
};


template<template<class ProcedureInfo> class TInternalQueueing, unsigned int TNumQueues>
struct BucketPriorityQueueTypingTakeOptimized
{
	template<class ProcedureInfo>
	class Type : public BucketPriorityQueue <ProcedureInfo, TInternalQueueing, TNumQueues, BucketLinearPriorityFast> {};
};
