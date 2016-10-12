#pragma once
#include "queueInterface.cuh"
#include "procinfoTemplate.cuh"
#include "common.cuh"
#include "random.cuh"

template<class TProcInfo, template<class /*ProcInfo*/> class Q>
class EnqueueSelectedQueue;

template<class TProcInfo, template <class /*ProcInfo*/> class Q, class ... ThisProcedures>
struct CombinerQueueElement
{
	typedef SubProcInfo<TProcInfo, ThisProcedures...> Procedures;
	template<class ProcInfo>
	struct Queue : public Q<ProcInfo> {};
};

template<class TProcInfo, template<class, int >  class DequeueSelector, class ... CombinerQueueElements>
class CombinerQueueInternal;


template<class TProcInfo, template<class, int > class DequeueSelector, class ... CombinerQueueElements>
class CombinerQueue : public CombinerQueueInternal<TProcInfo, DequeueSelector, CombinerQueueElements...>
{
	template<bool MultiProcedure>
	struct DequeueVisitor
	{
		void*& data; 
		int*& procId;  
		int maxShared;
		int result;
		__inline__ __device__ DequeueVisitor(void*& data, int*& procId, int maxShared) : data(data), procId(procId), maxShared(maxShared)
		{
		}
		template<class Q>
		__inline__ __device__ bool complete(Q& q)
		{
			result = q. template dequeue<MultiProcedure>(data, procId, maxShared);
			if (result > 0)
				return true;
			return false;
		}
	};
	template<bool MultiProcedure>
	struct DequeueStartReadVisitor
	{
		void*& data;
		int*& procId;
		int maxShared;
		int result;
		__inline__ __device__ DequeueStartReadVisitor(void*& data, int*& procId, int maxShared) : data(data), procId(procId), maxShared(maxShared)
		{
		}
		template<class Q>
		__inline__ __device__ bool complete(Q& q)
		{
			result = q. template dequeueStartRead<MultiProcedure>(data, procId, maxShared);
			if (result > 0)
				return true;
			return false;
		}
	};
public:
	template<bool MultiProcedure>
	__inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = -1)
	{
		DequeueSelector<DequeueVisitor<MultiProcedure>, sizeof...(CombinerQueueElements)> visitor(data, procId, maxShared);
		do
		{
			CombinerQueueInternal<TProcInfo, DequeueSelector, CombinerQueueElements...> :: template visit<0>(visitor);
		} while (visitor.revisit());
		return visitor.get().result;
	}


	template<bool MultiProcedure>
	__inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = -1)
	{
		DequeueSelector<DequeueStartReadVisitor<MultiProcedure>, sizeof...(CombinerQueueElements)> visitor(data, procId, maxShared);
		do
		{
			CombinerQueueInternal<TProcInfo, DequeueSelector, CombinerQueueElements...> :: template visit<0> (visitor);
		} while (visitor.revisit());
		return visitor.get().result;
	}
};

template<class TProcInfo, template<class, int > class DequeueSelector, class FirstElement, class ... RemainingElements>
class CombinerQueueInternal<TProcInfo, DequeueSelector, FirstElement, RemainingElements ...> 
{
	typedef FirstElement::Queue<typename FirstElement::Procedures> ThisQueue;
	typedef CombinerQueueInternal<TProcInfo, DequeueSelector, RemainingElements ...> NextCombinerQueue;

	ThisQueue thisQueue;
	NextCombinerQueue nextCombinerQueue;

public:
	static const bool needTripleCall = false; /*<- TODO: not so easy to unify...*/
	static const bool supportReuseInit = ThisQueue::supportReuseInit || NextCombinerQueue::supportReuseInit;
	static const int globalMaintainMinThreads = maxOperator<ThisQueue::globalMaintainMinThreads, NextCombinerQueue::globalMaintainMinThreads>::result;

	static int globalMaintainSharedMemory(int Threads) {
		return std::max(ThisQueue::globalMaintainSharedMemory(Threads), NextCombinerQueue::globalMaintainSharedMemory(Threads));
	}
	static const int requiredShared = maxOperator< ThisQueue::requiredShared, NextCombinerQueue::requiredShared> ::result;

	__inline__ __device__ void init()
	{
		thisQueue.init();
		nextCombinerQueue.init();
	}

	template<class PROCEDURE>
	__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			return thisQueue . template enqueueInitial<PROCEDURE>(data);
		return nextCombinerQueue. template enqueueInitial<PROCEDURE>(data);
	}

	template<int threads, class PROCEDURE>
	__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData* data)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			return thisQueue . template enqueueInitial<threads, PROCEDURE>(data);
		return nextCombinerQueue. template enqueueInitial<threads, PROCEDURE>(data);
	}

	template<class PROCEDURE>
	__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			return thisQueue. template enqueue<PROCEDURE>(data);
		return nextCombinerQueue. template enqueue<PROCEDURE>(data);
	}

	template<int threads, class PROCEDURE>
	__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			return thisQueue. template enqueue<threads, PROCEDURE>(data);
		return nextCombinerQueue. template enqueue<threads, PROCEDURE>(data);
	}


	template<bool MultiProcedure>
	__inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
	{
		int res = 0;
		if ((res = thisQueue. template dequeueSelected<MultiProcedure>(data, procId, maxNum))) return res;
		return nextCombinerQueue. template dequeueSelected<MultiProcedure>(data, procId, maxNum);
	}

	template<class PROCEDURE>
	__inline__ __device__ int reserveRead(int maxNum = -1)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			return thisQueue. template dequeueSelected<PROCEDURE>(maxNum);
		return nextCombinerQueue. template dequeueSelected<PROCEDURE>(maxNum);
	}
	template<class PROCEDURE>
	__inline__ __device__ int startRead(void*& data, int num)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			return thisQueue. template startRead<PROCEDURE>(maxNum);
		return nextCombinerQueue. template startRead<PROCEDURE>(maxNum);
	}
	template<class PROCEDURE>
	__inline__ __device__ void finishRead(int id, int num)
	{
		if (FirstElement::Procedures:: template Contains<PROCEDURE>::value)
			thisQueue. template finishRead<PROCEDURE>(id, num);
		nextCombinerQueue. template finishRead<PROCEDURE>(id, num);
	}

	__inline__ __device__ void numEntries(int* counts)
	{
		thisQueue.numEntries(counts);
		nextCombinerQueue.numEntries(counts);
	}


	__inline__ __device__ void record()
	{
		thisQueue.record();
		nextCombinerQueue.record();
	}
	__inline__ __device__ void reset()
	{
		thisQueue.reset();
		nextCombinerQueue.reset();
	}


	__inline__ __device__ void workerStart()
	{
		thisQueue.workerStart();
		nextCombinerQueue.workerStart();
	}
	__inline__ __device__ void workerMaintain()
	{
		thisQueue.workerMaintain();
		nextCombinerQueue.workerMaintain();
	}
	__inline__ __device__ void workerEnd()
	{
		thisQueue.workerEnd();
		nextCombinerQueue.workerEnd();
	}
	__inline__ __device__ void globalMaintain()
	{
		thisQueue.globalMaintain();
		nextCombinerQueue.globalMaintain();
	}

	static std::string name()
	{
		return ThisQueue::name() + NextCombinerQueue::name();
	}

	template<int N, class Visitor>
	__inline__ __device__ void visit(Visitor& visitor)
	{
		if (!visitor. template visit<typename FirstElement::Procedures, N>(thisQueue))
			nextCombinerQueue. template visit<N+1>(visitor);
	}
};


template<class TProcInfo, template<class, int > class DequeueSelector>
class CombinerQueueInternal<TProcInfo, DequeueSelector>
{

public:
	static const bool needTripleCall = false; /*<- TODO: not so easy to unify...*/
	static const bool supportReuseInit = false;
	static const int globalMaintainMinThreads = 0;

	static int globalMaintainSharedMemory(int Threads) {
		return 0;
	}
	static const int requiredShared = 0;

	__inline__ __device__ void init()
	{
	}

	template<class PROCEDURE>
	__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data)
	{
		return false;
	}

	template<int threads, class PROCEDURE>
	__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData* data)
	{
		return false;
	}

	template<class PROCEDURE>
	__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data)
	{
		return false;
	}

	template<int threads, class PROCEDURE>
	__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data)
	{
		return false;
	}


	template<bool MultiProcedure>
	__inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
	{
		return 0;
	}

	template<class PROCEDURE>
	__inline__ __device__ int reserveRead(int maxNum = -1)
	{
		return 0;
	}
	template<class PROCEDURE>
	__inline__ __device__ int startRead(void*& data, int num)
	{
		return 0;
	}
	template<class PROCEDURE>
	__inline__ __device__ void finishRead(int id, int num)
	{
	}

	__inline__ __device__ void numEntries(int* counts)
	{
	}
	__inline__ __device__ void record()
	{
	}
	__inline__ __device__ void reset()
	{
	}
	__inline__ __device__ void workerStart()
	{
	}
	__inline__ __device__ void workerMaintain()
	{
	}
	__inline__ __device__ void workerEnd()
	{
	}
	__inline__ __device__ void globalMaintain()
	{
	}

	static std::string name()
	{
		return "";
	}
	template<int N, class Visitor>
	__inline__ __device__ void visit(Visitor& visitor)
	{
	}
};


template<class Handler, int NumQueues>
class DequeueFirstFirstSelector
{
	Handler handler;
public:
	
	template<class ... Arguments>
	__inline__ __device__ DequeueFirstFirstSelector(Arguments& ... args) : handler(args...) {}

	__inline__ __device__ Handler& get()
	{
		return handler;
	}

	__inline__ __device__ bool revisit()
	{
		return false;
	}

	template<class ProcInfo, int N, class Q>
	__inline__ __device__ bool visit(Q& q)
	{
		return handler.complete(q);
	}
};

template<class Handler, int NumQueues>
class DequeueRoundRobin
{
	Handler handler;
	int sel;
	bool completed;
public:

	template<class ... Arguments>
	__inline__ __device__ DequeueRoundRobin(Arguments &... args) : handler(args...), completed(false)
	{
		__shared__ int lastSelection;
		if (threadIdx.x == 0)
		{
			lastSelection = (lastSelection + 1) % NumQueues;
		}
		__syncthreads();
		sel = lastSelection;
	}

	__inline__ __device__ Handler& get()
	{
		return handler;
	}
	template<class ProcInfo, int N, class Q>
	__inline__ __device__ bool visit(Q& q)
	{
		if (N >= sel || -N > sel)
			return (completed = handler.complete(q));
		return false;
	}
	__inline__ __device__ bool revisit()
	{
		if (!completed && sel > 0)
		{
			sel = -sel;
			return true;
		}
		return false;
	}      
};

template<int NegativeEvals, int ... Probs>
struct EvalProbabilities;

template<int FirstP, int ... OtherProbs>
struct EvalProbabilities<0, FirstP, OtherProbs...>
{
	static __inline__ __device__ bool eval()
	{
		return qrandom::block_check(FirstP );
	}
};

template<int ... OtherProbs>
struct EvalProbabilities<0, 0, OtherProbs...>
{
	static __inline__ __device__ bool eval()
	{
		return false;
	}
};

template<int FirstP>
struct EvalProbabilities<0, FirstP>
{
	static __inline__ __device__ bool eval()
	{
		return true;
	}
};

template<>
struct EvalProbabilities<0, 0>
{
	static __inline__ __device__ bool eval()
	{
		return false;
	}
};

template<int NegativeEvals, int FirstP, int ... OtherProbs>
struct EvalProbabilities<NegativeEvals, FirstP, OtherProbs...>
{
	static __inline__ __device__ bool eval()
	{
		return EvalProbabilities<NegativeEvals - 1, ((FirstP < 100) ? (OtherProbs * 100 / (100 - FirstP)) : 0)... >::eval();
	}
};

template<int NegativeEvals, int ... OtherProbs>
struct EvalProbabilities<NegativeEvals, 0, OtherProbs...>
{
	static __inline__ __device__ bool eval()
	{
		return EvalProbabilities<NegativeEvals - 1, OtherProbs...>::eval();
	}
};

template<int ... Probabilities>
class DequeueProbabilityConfigurator
{
public:
	template<class Handler, int NumQueues>
	class Selector
	{
		Handler handler;
		bool completed;
	public:

		template<class ... Arguments>
		__inline__ __device__ Selector(Arguments & ... args) : handler(args...), completed(false) {}

		__inline__ __device__ Handler& get()
		{
			return handler;
		}

		__inline__ __device__ bool revisit()
		{
			if (completed)
				return false;
			completed = true;
			return true;
		}

		template<class ProcInfo, int N, class Q>
		__inline__ __device__ bool visit(Q& q)
		{
			if (completed)
				return handler.complete(q);
			else if (EvalProbabilities<N, Probabilities...>::eval())
				return (completed = handler.complete(q));
			return false;
		}
	};
};